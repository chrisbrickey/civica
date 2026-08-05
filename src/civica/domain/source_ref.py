"""SourceRef domain model: a pointer to a corpus location backing a claim."""

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):  # type: ignore[explicit-any]
    """An immutable reference to a page and section in the corpus."""

    model_config = ConfigDict(frozen=True)

    page_slug: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
