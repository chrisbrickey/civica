"""The user identifier that links a learner's records across modules."""

from typing import NewType
from uuid import UUID

UserId = NewType("UserId", UUID)
