from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator


class RubricPoint(BaseModel):
    point_id: str = Field(..., description="Unique rubric point id, e.g. P1")
    desc: str = Field(..., description="Rubric point description")
    score: float = Field(..., ge=0.0)


class GradingTaskInput(BaseModel):
    question_id: str
    question: str
    reference_answer: str
    rubric: List[RubricPoint]
    student_answer: str

    @model_validator(mode="after")
    def validate_unique_point_id(self) -> "GradingTaskInput":
        point_ids = [point.point_id for point in self.rubric]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("rubric point_id values must be unique")
        return self


class PointJudgement(BaseModel):
    point_id: str
    triggered: bool
    evidence: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)


class GradingOutput(BaseModel):
    triggered_points: List[PointJudgement]
    final_score: float

    @model_validator(mode="after")
    def validate_unique_triggered_point_id(self) -> "GradingOutput":
        point_ids = [point.point_id for point in self.triggered_points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("triggered_points point_id values must be unique")
        return self


class SFTSample(BaseModel):
    question_id: str
    prompt: str
    target: str
    metadata: Dict = Field(default_factory=dict)
