"""Shared synthetic learner ids for tests."""

import uuid

from civica.domain.user import UserId

USER_ID = UserId(uuid.UUID("00000000-0000-0000-0000-000000000001"))
OTHER_USER_ID = UserId(uuid.UUID("00000000-0000-0000-0000-000000000002"))

# Minimum (as of bycrpyt 5.0.0) injected encryption rounds so tests that hash secrets run as quickly as possible
TEST_BCRYPT_ROUNDS = 4
