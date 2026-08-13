from __future__ import annotations

from enum import Enum
from typing import List, Literal, Union, Annotated, Optional, Dict
from pydantic import BaseModel, Field, model_validator


class TriScore(str, Enum):
    YES = "YES"
    NO = "NO"
    NULL = "NULL"


class YesNo(str, Enum):
    YES = "YES"
    NO = "NO"


class Observability(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"


class ObservabilityDerived(str, Enum):
    DERIVED = "DERIVED"


class ProficiencyScore(str, Enum):
    PROFICIENT = "PROFICIENT"
    NOT_PROFICIENT = "NOT_PROFICIENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FlowOrganization(str, Enum):
    organized = "organized"
    mixed = "mixed"
    disorganized = "disorganized"
    unknown = "unknown"


class WastedMotionType(str, Enum):
    instrument_search = "instrument_search"
    pause_reset = "pause_reset"
    failed_pass_sequence = "failed_pass_sequence"
    excess_regrasp_cluster = "excess_regrasp_cluster"
    other = "other"


class CoachingTag(str, Enum):
    ARTERIOTOMY_TOO_SHORT = "ARTERIOTOMY_TOO_SHORT"
    SPATULATION_CONCAVE_GEOMETRY = "SPATULATION_CONCAVE_GEOMETRY"
    SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS = "SUTURE_GRASPED_WITH_FORCEPS_OR_PICKUPS"
    TIE_DIRECTLY_ON_TOE = "TIE_DIRECTLY_ON_TOE"
    NEEDLE_DIRECTION_REVERSED = "NEEDLE_DIRECTION_REVERSED"
    ASSISTANT_QUALITY_IMPACT = "ASSISTANT_QUALITY_IMPACT"
    SPACING_TOO_CLOSE = "SPACING_TOO_CLOSE"
    SPACING_CAN_BE_WIDER = "SPACING_CAN_BE_WIDER"


class ProjectionMode(str, Enum):
    construct_mode = "construct_mode"
    hawk_mode = "hawk_mode"
    dove_mode = "dove_mode"


class WastedMotionEvent(BaseModel):
    type: WastedMotionType
    count_estimate: Annotated[float, Field(ge=0)]
    note: str


class Item6aSubitem(BaseModel):
    score: TriScore
    evidence: str


class Item6bSubitem(BaseModel):
    score: TriScore
    evidence: str


class Item6Subitems(BaseModel):
    six_a_right_angle_method: Item6aSubitem = Field(alias="6a_right_angle_method")
    six_b_safe_transfer_outcome: Item6bSubitem = Field(alias="6b_safe_transfer_outcome")

    model_config = {"populate_by_name": True}


class EvidenceBasedItem(BaseModel):
    item_id: Annotated[int, Field(ge=1, le=10)]
    score: TriScore
    observability: Observability
    evidence: str
    coaching: Optional[str] = None
    subitems: Optional[Item6Subitems] = None


class EconomyEvidence(BaseModel):
    flow_organization: FlowOrganization
    wasted_motion_events: List[WastedMotionEvent] = Field(default_factory=list, max_length=10)
    economy_index: Optional[float] = None


class EconomyItem(BaseModel):
    item_id: Literal[11] = 11
    score: Union[Annotated[int, Field(ge=1, le=5)], Literal["NULL"]]
    observability: Observability
    coaching: Optional[str] = None
    evidence: EconomyEvidence


class ProficiencyEvidence(BaseModel):
    red_lines_triggered: List[str] = Field(default_factory=list)
    missing_core_domains: List[str] = Field(default_factory=list)


class ProficiencyItem(BaseModel):
    item_id: Literal[12] = 12
    score: ProficiencyScore
    observability: Literal["DERIVED"] = "DERIVED"
    evidence: ProficiencyEvidence


class CommentsEvidence(BaseModel):
    coaching_tags: List[CoachingTag] = Field(default_factory=list, max_length=5)


class CommentsItem(BaseModel):
    item_id: Literal[13] = 13
    score: str
    observability: Literal["OBSERVED"] = "OBSERVED"
    evidence: CommentsEvidence


class CoverageInfo(BaseModel):
    observed_count: int
    total_count: Literal[10] = 10
    observed_percent: float
    core_observed: Dict[str, bool]


class ProjectedDocx(BaseModel):
    mode: ProjectionMode
    items_1_10: Dict[str, str]
    economy_1_5: int
    proficiency_yes_no: str
    projection_notes: List[str] = Field(default_factory=list)


EvidenceBasedItemUnion = Union[EvidenceBasedItem, EconomyItem, ProficiencyItem, CommentsItem]


class EvidenceBasedOutput(BaseModel):
    items: List[EvidenceBasedItemUnion]
    coverage: CoverageInfo


class VopModeratedV1Output(BaseModel):
    case_id: str
    rubric_version: Literal["vop_2023_v1"] = "vop_2023_v1"
    evidence_based: EvidenceBasedOutput
    projected_docx: Optional[ProjectedDocx] = None

    @model_validator(mode="after")
    def validate_items(self) -> VopModeratedV1Output:
        items = self.evidence_based.items
        if len(items) != 13:
            raise ValueError("items must contain exactly 13 elements")

        ids = [it.item_id for it in items]
        if set(ids) != set(range(1, 14)):
            raise ValueError("items must include exactly one of each item_id 1..13")

        if len(set(ids)) != 13:
            raise ValueError("duplicate item_id values found")

        return self


class VopDocxV41Output(BaseModel):
    case_id: str
    rubric_version: Literal["vop_docx_v4_1"] = "vop_docx_v4_1"
    policies: Dict[str, Union[bool, str]]
    items: List[Dict]

    @model_validator(mode="after")
    def validate_items(self) -> VopDocxV41Output:
        if len(self.items) != 13:
            raise ValueError("items must contain exactly 13 elements")

        ids = [it.get("item_id") if isinstance(it, dict) else getattr(it, "item_id", None) for it in self.items]
        if set(ids) != set(range(1, 14)):
            raise ValueError("items must include exactly one of each item_id 1..13")

        if len(set(ids)) != 13:
            raise ValueError("duplicate item_id values found")

        return self
