"""Username and password rules checked at registration. NIST SP 800-63B for passwords."""

import re
from functools import cache
from importlib.resources import files

USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32
RESERVED_USERNAMES = frozenset({"admin", "root", "system", "civica", "support"})

# Shortest secret accepted in characters.
MIN_SECRET_LENGTH = 8

# bcrypt silently truncates (or rejects in new releases) secrets whose UTF-8 encoding exceeds this many bytes.
# Reject over-length secrets with a clear error instead of hashing a truncated value.
MAX_SECRET_BYTES = 72

USERNAME_RULES = (
    f"{USERNAME_MIN_LENGTH} to {USERNAME_MAX_LENGTH} characters: lowercase letters, digits, "
    "'.', '_' or '-', starting and ending with a letter or digit"
)
SECRET_RULES = f"{MIN_SECRET_LENGTH} to {MAX_SECRET_BYTES} characters"

# Lowercase ASCII letters and digits at both ends, separators allowed in between.
_USERNAME_PATTERN = re.compile(r"[a-z0-9]([a-z0-9._-]*[a-z0-9])?")


class InvalidUsername(ValueError):
    """Raised when a username breaks the registration rules; the message is safe to show users."""


class InvalidSecret(ValueError):
    """Raised when a password breaks the registration rules; the message is safe to show users."""


@cache
def _common_secrets() -> frozenset[str]:
    text = files("civica.users").joinpath("common_passwords.txt").read_text(encoding="utf-8")
    lines = (line.strip() for line in text.splitlines())
    return frozenset(line.casefold() for line in lines if line and not line.startswith("#"))


def validate_username(username: str) -> None:
    """Check an already-normalized username against the registration rules."""
    length_ok = USERNAME_MIN_LENGTH <= len(username) <= USERNAME_MAX_LENGTH
    if not length_ok or _USERNAME_PATTERN.fullmatch(username) is None:
        raise InvalidUsername(f"Username must be {USERNAME_RULES}.")
    if username in RESERVED_USERNAMES:
        raise InvalidUsername("That username is reserved. Choose another.")


def validate_secret(secret: str, username: str) -> None:
    """Check a password against the registration rules, given the normalized username."""
    if len(secret) < MIN_SECRET_LENGTH or len(secret.encode("utf-8")) > MAX_SECRET_BYTES:
        raise InvalidSecret(f"Password must be {SECRET_RULES}.")
    if secret.casefold() in _common_secrets():
        raise InvalidSecret("That password is too common. Choose another.")
    if secret.casefold() == username:
        raise InvalidSecret("Password must not match the username.")
