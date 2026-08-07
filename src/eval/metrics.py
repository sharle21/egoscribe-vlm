"""Metric computation for the EgoScribe eval harness (PRD section 8).

Kept separate from evaluate.py so the scoring logic is unit-testable without a GPU or model.
All metrics derive from HandObjectInteraction schema fields:

  - schema validity rate   : did the raw generation parse into a valid HandObjectInteraction?
  - free-text fields        : tool_detected, target_object, action_verb, current_state
                              -> normalized exact-match AND token-F1 (soft overlap). Exact-match
                                 is the honest-but-harsh headline; token-F1 is the soft signal.
                                 Both reported because labels are weak (Claude-bootstrapped), so
                                 a low exact-match is expected and not by itself a model failure.
  - point_of_no_return      : binary F1 (positive class = True)
  - safety_gear_missing     : set-level precision / recall / F1 over the union of all gear items

Only records whose prediction parsed contribute to field-level metrics; the validity rate
captures the rest so a model can't win by emitting garbage that dodges scoring.
"""
import random
import re
import string
from collections import Counter, defaultdict

FREE_TEXT_FIELDS = ["tool_detected", "target_object", "action_verb", "current_state"]

# Map punctuation -> space (not delete) so "circuit-board" becomes two tokens, not "circuitboard".
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(val):
    """Lowercase, punctuation->space, collapse whitespace. None -> '' (so null==null matches)."""
    if val is None:
        return ""
    s = str(val).lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred, gold):
    """Bag-of-words F1 between two strings (SQuAD-style). 1.0 if both empty."""
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tokens)
    recall = n_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _scalar_scores(preds, golds):
    """The headline scalar metrics, recomputed from scratch on any subset of records. Single
    source of truth for both the point estimate and the take-level bootstrap, so a resampled CI
    can never disagree with the point estimate on how a number is defined."""
    n = len(preds)
    if n == 0:
        return {}
    scores = {}
    for field in FREE_TEXT_FIELDS:
        scores[f"{field}.exact_match"] = sum(
            normalize_text(p.get(field)) == normalize_text(g.get(field)) for p, g in zip(preds, golds)) / n
        scores[f"{field}.token_f1"] = sum(token_f1(p.get(field), g.get(field)) for p, g in zip(preds, golds)) / n

    tp = fp = fn = correct = 0
    for p, g in zip(preds, golds):
        pv, gv = bool(p.get("point_of_no_return_detected")), bool(g.get("point_of_no_return_detected"))
        tp += pv and gv
        fp += pv and not gv
        fn += (not pv) and gv
        correct += pv == gv
    scores["point_of_no_return_detected.f1"] = _prf(tp, fp, fn)["f1"]
    scores["point_of_no_return_detected.accuracy"] = correct / n

    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        pset = {normalize_text(x) for x in (p.get("safety_gear_missing") or []) if normalize_text(x)}
        gset = {normalize_text(x) for x in (g.get("safety_gear_missing") or []) if normalize_text(x)}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    scores["safety_gear_missing.f1"] = _prf(tp, fp, fn)["f1"]
    return scores


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in [0,100]) over an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100) * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def take_level_bootstrap(preds, golds, take_ids, n_boot=1000, seed=0):
    """Block bootstrap over TAKES (the real statistical unit — segments within a take are
    correlated). Resamples takes with replacement, pools their segments, recomputes every headline
    scalar, and returns 95% percentile CIs. With only ~6 held-out takes these CIs are wide by
    construction: an honesty instrument that stops small A-vs-D deltas being read as significant,
    not a significance test."""
    idx_by_take = defaultdict(list)
    for i, t in enumerate(take_ids):
        idx_by_take[t].append(i)
    takes = list(idx_by_take)
    if len(takes) < 2:
        return {"n_takes": len(takes), "note": "too few takes to bootstrap"}

    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        idxs = []
        for _ in takes:  # resample len(takes) takes, with replacement
            idxs.extend(idx_by_take[rng.choice(takes)])
        for k, v in _scalar_scores([preds[i] for i in idxs], [golds[i] for i in idxs]).items():
            samples[k].append(v)

    ci = {}
    for k, vals in samples.items():
        vals.sort()
        ci[k] = [round(_percentile(vals, 2.5), 4), round(_percentile(vals, 97.5), 4)]
    return {"n_takes": len(takes), "n_boot": n_boot, "ci_95": ci,
            "note": "95% CI from block bootstrap over takes; wide at this n by design."}


def compute_metrics(preds, golds, num_total, take_ids=None, n_boot=1000, seed=0):
    """preds/golds: lists of parsed dicts (same length, only successfully-parsed records).
    num_total: total eval records attempted (>= len(preds)), for the validity rate.
    take_ids: optional list (parallel to preds/golds) of the take each record belongs to. When
        given, adds take-level bootstrap CIs for the headline scalars — REQUIRED for any honest
        cross-strategy comparison (see ADR-0002 interpretation section).
    """
    n = len(preds)
    metrics = {
        "num_total": num_total,
        "num_parsed": n,
        "schema_validity_rate": n / num_total if num_total else 0.0,
    }

    # Free-text: normalized exact-match + mean token-F1.
    for field in FREE_TEXT_FIELDS:
        exact = sum(normalize_text(p.get(field)) == normalize_text(g.get(field)) for p, g in zip(preds, golds))
        tf1 = sum(token_f1(p.get(field), g.get(field)) for p, g in zip(preds, golds))
        metrics[field] = {
            "exact_match": exact / n if n else 0.0,
            "token_f1": tf1 / n if n else 0.0,
        }

    # point_of_no_return_detected: binary F1, positive class = True.
    tp = fp = fn = tn = 0
    for p, g in zip(preds, golds):
        pv, gv = bool(p.get("point_of_no_return_detected")), bool(g.get("point_of_no_return_detected"))
        tp += pv and gv
        fp += pv and not gv
        fn += (not pv) and gv
        tn += (not pv) and (not gv)
    ponr = _prf(tp, fp, fn)
    ponr.update({"accuracy": (tp + tn) / n if n else 0.0, "support_positive": tp + fn})
    metrics["point_of_no_return_detected"] = ponr

    # safety_gear_missing: micro set-level P/R/F1 over normalized gear items.
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        pset = {normalize_text(x) for x in (p.get("safety_gear_missing") or []) if normalize_text(x)}
        gset = {normalize_text(x) for x in (g.get("safety_gear_missing") or []) if normalize_text(x)}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    gear = _prf(tp, fp, fn)
    gear["support_gold_items"] = tp + fn
    metrics["safety_gear_missing"] = gear

    if take_ids is not None:
        metrics["bootstrap"] = take_level_bootstrap(preds, golds, take_ids, n_boot=n_boot, seed=seed)

    return metrics
