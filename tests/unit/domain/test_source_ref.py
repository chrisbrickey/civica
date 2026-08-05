"""Unit tests for civica.domain.source_ref: SourceRef pydantic domain model."""

from typing import Any

import pytest

from civica.domain.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SOURCE_REF_FIELDS: dict[str, Any] = {
    "page_slug": "test-page",
    "section_id": "sec_001",
}


def make_source_ref(**overrides: Any) -> SourceRef:
    """Build a SourceRef from the shared valid fields, overriding a subset."""
    return SourceRef(**{**VALID_SOURCE_REF_FIELDS, **overrides})


# ---------------------------------------------------------------------------
# Validation: invalid field values rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("page_slug", ""),
        ("section_id", ""),
    ],
)
def test_invalid_field_value_raises_validation_error(
    field: str, invalid_value: Any
) -> None:
    with pytest.raises(ValueError):
        make_source_ref(**{field: invalid_value})


# ---------------------------------------------------------------------------
# Frozen model: mutation raises
# ---------------------------------------------------------------------------


def test_frozen_source_ref_raises_on_mutation() -> None:
    source_ref = make_source_ref()
    with pytest.raises(ValueError):
        source_ref.page_slug = "mutated-page"
