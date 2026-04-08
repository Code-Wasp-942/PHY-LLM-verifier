from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class VerifyBenchSample(BaseModel):
    question_id: str
    completion_id: str | None = None
    question: str
    answer: str
    completion: str | None = None
    gold_correct: bool | None = None
    completion_model: str | None = None
    source: str | None = None
    answer_type: str | None = None
    answer_subtype: str | None = None
    annotator: List[str] = Field(default_factory=list)


class SFTSample(BaseModel):
    question_id: str
    prompt: str
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
