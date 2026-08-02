"""Chunk domain model: a single unit of corpus content."""

from pydantic import BaseModel, ConfigDict, Field

from civica.domain.themes import Theme


class Chunk(BaseModel):  # type: ignore[explicit-any]
    """An immutable slice of corpus
    tied to a theme, page, and section
    (per the official study materials).
    """

    model_config = ConfigDict(frozen=True)

    theme: Theme
    page_slug: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(min_length=1)
    text: str = Field(min_length=1)
