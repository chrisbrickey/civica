"""Fixtures shared by the assessment unit tests."""

import pytest

from tests.support.assessment import FakeAnswerLogger


@pytest.fixture()
def answer_logger() -> FakeAnswerLogger:
    return FakeAnswerLogger()
