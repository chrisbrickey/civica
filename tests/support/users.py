"""Shared synthetic learner ids for tests."""

import uuid

from civica.domain.user import UserId

USER_ID = UserId(uuid.UUID("00000000-0000-0000-0000-000000000001"))
OTHER_USER_ID = UserId(uuid.UUID("00000000-0000-0000-0000-000000000002"))
