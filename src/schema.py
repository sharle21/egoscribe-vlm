from pydantic import BaseModel, Field
from typing import Optional, List

class HandObjectInteraction(BaseModel):
    tool_detected: Optional[str] = Field(None, description="The tool being held by the user, e.g., 'screwdriver'")
    target_object: str = Field(..., description="The main object being manipulated, e.g., 'circuit_board'")
    action_verb: str = Field(..., description="The precise action being taken, e.g., 'unscrewing', 'soldering'")
    
    # State Change Analysis
    point_of_no_return_detected: bool = Field(
        ..., description="True if the exact moment of physical state change occurred in this frame window"
    )
    current_state: str = Field(..., description="Current structural state, e.g., 'disassembled', 'secured'")
    
    # Operational Safeguards
    safety_gear_missing: List[str] = Field(
        default_factory=list, description="List of missing safety elements, e.g., ['gloves', 'goggles']"
    )