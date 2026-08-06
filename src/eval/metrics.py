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
import re
import string
from collections import Counter

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


def compute_metrics(preds, golds, num_total):
    """preds/golds: lists of parsed dicts (same length, only successfully-parsed records).
    num_total: total eval records attempted (>= len(preds)), for the validity rate.
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

    return metrics
