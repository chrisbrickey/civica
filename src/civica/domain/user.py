"""User domain model: identifies a learner."""

from typing import NewType
from uuid import UUID

UserId = NewType("UserId", UUID)
