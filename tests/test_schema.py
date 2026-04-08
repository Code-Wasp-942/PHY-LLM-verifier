from pydantic import ValidationError

from src.schema import VerifyBenchSample


def test_verifybench_sample_requires_question_and_answer() -> None:
    sample = VerifyBenchSample(
        question_id="Q1",
        question="What is 1+1?",
        answer="2",
    )
    assert sample.question_id == "Q1"
    assert sample.answer == "2"


def test_verifybench_sample_missing_answer_fails() -> None:
    try:
        VerifyBenchSample(question_id="Q2", question="q")
        assert False, "ValidationError expected"
    except ValidationError:
        assert True
