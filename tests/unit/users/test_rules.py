"""Unit tests for civica.users.rules: username and password rules checked at registration."""

import pytest

from civica.users.rules import (
    InvalidSecret,
    InvalidUsername,
    validate_secret,
    validate_username,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USERNAME = "test-user"
_SECRET = "sample-secret"

_USERNAME_RULES_TEXT = "3 to 32 characters"
_RESERVED_TEXT = "reserved"
_SECRET_RULES_TEXT = "8 to 72 characters"
_COMMON_SECRET_TEXT = "too common"
_SAME_AS_USERNAME_TEXT = "must not match the username"


# ---------------------------------------------------------------------------
# Username
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "username",
    ["abc", "a" * 32, "test.user_01", "test-user", "123"],
    ids=["min-length", "max-length", "dot-and-underscore", "hyphen", "digits-only"],
)
def test_valid_username_passes(username: str) -> None:
    validate_username(username)


@pytest.mark.parametrize(
    "username",
    [
        "ab",
        "a" * 33,
        "test user",
        "test@user",
        "tést-user",
        "test\nuser",
        "-test-user",
        "test-user.",
        "_test-user",
    ],
    ids=[
        "too-short",
        "too-long",
        "space",
        "symbol",
        "non-ascii-letter",
        "line-break",
        "leading-separator",
        "trailing-separator",
        "leading-underscore",
    ],
)
def test_invalid_username_raises_with_rules(username: str) -> None:
    with pytest.raises(InvalidUsername, match=_USERNAME_RULES_TEXT):
        validate_username(username)


@pytest.mark.parametrize("username", ["admin", "root", "system", "civica", "support"])
def test_reserved_username_raises(username: str) -> None:
    with pytest.raises(InvalidUsername, match=_RESERVED_TEXT):
        validate_username(username)


def test_invalid_username_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        validate_username("ab")


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    ["sample-8", "b" * 72, "sample secret with spaces", "été-secret"],
    ids=["min-length", "max-bytes", "spaces", "accented"],
)
def test_valid_secret_passes(secret: str) -> None:
    validate_secret(secret, _USERNAME)


@pytest.mark.parametrize(
    "secret",
    ["sample7", "b" * 73, "é" * 37],
    ids=["too-short", "too-many-bytes", "too-many-multibyte-chars"],
)
def test_secret_outside_length_limits_raises_with_rules(secret: str) -> None:
    with pytest.raises(InvalidSecret, match=_SECRET_RULES_TEXT):
        validate_secret(secret, _USERNAME)


@pytest.mark.parametrize("secret", ["password", "12345678", "PASSWORD", "qwerty123"])
def test_common_secret_raises(secret: str) -> None:
    with pytest.raises(InvalidSecret, match=_COMMON_SECRET_TEXT):
        validate_secret(secret, _USERNAME)


@pytest.mark.parametrize("secret", ["sample-username", "SAMPLE-USERNAME"], ids=["same", "case-variant"])
def test_secret_matching_username_raises(secret: str) -> None:
    with pytest.raises(InvalidSecret, match=_SAME_AS_USERNAME_TEXT):
        validate_secret(secret, "sample-username")


def test_invalid_secret_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        validate_secret("sample7", _USERNAME)
