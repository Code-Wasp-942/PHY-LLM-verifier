from pydantic import ValidationError

from src.schema import GradingOutput, GradingTaskInput


def test_input_requires_unique_rubric_point_ids() -> None:
    try:
        GradingTaskInput(
            question_id="Q1",
            question="q",
            reference_answer="a",
            rubric=[
                {"point_id": "P1", "desc": "d1", "score": 1.0},
                {"point_id": "P1", "desc": "d2", "score": 1.0},
            ],
            student_answer="s",
        )
        assert False, "ValidationError expected"
    except ValidationError:
        assert True


def test_output_requires_unique_triggered_point_ids() -> None:
    try:
        GradingOutput(
            triggered_points=[
                {"point_id": "P1", "triggered": True, "evidence": "x", "confidence": 0.9},
                {"point_id": "P1", "triggered": False, "evidence": "", "confidence": 0.1},
            ],
            final_score=1.0,
        )
        assert False, "ValidationError expected"
    except ValidationError:
        assert True
