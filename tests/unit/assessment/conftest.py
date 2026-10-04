"""Fixtures shared by the assessment unit tests."""

import pytest

from tests.support.assessment import RecordingAnswerLogger


@pytest.fixture()
def answer_logger() -> RecordingAnswerLogger:
    return RecordingAnswerLogger()
