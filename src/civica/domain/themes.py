"""
Five official civic education themes per https://formation-civique.interieur.gouv.fr/.
"""

from pydantic import BaseModel, ConfigDict, Field


class Theme(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    slug: str = Field(min_length=1)
    display_name_fr: str = Field(min_length=1)

    @classmethod
    def from_slug(cls, slug: str) -> "Theme":
        """Return the Theme whose slug matches, raising ValueError on miss."""
        if slug in THEMES_BY_SLUG:
            return THEMES_BY_SLUG[slug]
        valid = ", ".join(THEMES_BY_SLUG.keys())
        raise ValueError(
            f"Unknown theme slug: {slug!r}. Valid slugs: {valid}"
        )


PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE = Theme(
    slug="principes-et-valeurs-de-la-republique",
    display_name_fr="Principes et valeurs de la République",
)

DROITS_ET_DEVOIRS = Theme(
    slug="droits-et-devoirs",
    display_name_fr="Droits et devoirs",
)

HISTOIRE_GEOGRAPHIE_ET_CULTURE = Theme(
    slug="histoire-geographie-et-culture",
    display_name_fr="Histoire, géographie et culture",
)

SYSTEME_INSTITUTIONNEL_ET_POLITIQUE = Theme(
    slug="systeme-institutionnel-et-politique",
    display_name_fr="Système institutionnel et politique",
)

VIVRE_DANS_LA_SOCIETE_FRANCAISE = Theme(
    slug="vivre-dans-la-societe-francaise",
    display_name_fr="Vivre dans la société française",
)

THEMES_BY_SLUG: dict[str, Theme] = {
    t.slug: t
    for t in [
        PRINCIPES_ET_VALEURS_DE_LA_REPUBLIQUE,
        DROITS_ET_DEVOIRS,
        HISTOIRE_GEOGRAPHIE_ET_CULTURE,
        SYSTEME_INSTITUTIONNEL_ET_POLITIQUE,
        VIVRE_DANS_LA_SOCIETE_FRANCAISE,
    ]
}
