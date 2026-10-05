"""Every LLM prompt of the assessment pipeline, in one place."""

# Bump PROMPT_VERSION whenever any prompt in PROMPTS changes. Stored with every question and rejection.
PROMPT_VERSION = "quiz-v003"

PROMPTS: dict[str, str] = {
    "system": (
        "Tu es un concepteur de questions pour l'examen civique de naturalisation francaise. "
        "Utilise uniquement les passages officiels fournis, sans connaissances externes. "
        "Redige en francais des questions a choix multiples avec exactement 4 options, "
        "dont une seule est correcte. Les mauvaises options doivent etre plausibles mais fausses. "
        "Chaque question doit se comprendre seule : ne mentionne jamais le passage, le texte, la fiche ou le document. "
        "Interroge sur les connaissances civiques, jamais sur les fiches elles-memes (titres, objectifs, contenu du cours). "
        "Une question de connaissance porte sur un fait du passage. "
        "Une question de mise en situation decrit un cas concret a resoudre avec le passage. "
        "Reponds uniquement par un tableau JSON, sans texte autour, de la forme "
        '[{"text": "...", "options": ["...", "...", "...", "..."], "correct_index": 0}]. '
        "correct_index est l'indice (0 a 3) de la bonne option."
    ),
    "human": (
        "Theme : {theme}\n\n"
        "Ecris exactement {count} questions, une par groupe de passages numerote, dans l'ordre "
        "des groupes. Chaque question s'appuie uniquement sur son groupe.\n\n{groups}"
    ),
    "critic_system": (
        "Tu es un verificateur de questions pour l'examen civique de naturalisation francaise. "
        "Pour chaque question, juge-la uniquement a partir des passages officiels fournis, "
        "sans connaissances externes. Une question est acceptee seulement si : "
        "1) exactement une option est correcte ; "
        "2) la bonne option indiquee est bien celle que les passages confirment ; "
        "3) chaque option et la reponse indiquee s'appuient sur les passages, sans fait invente ; "
        "4) les mauvaises options sont plausibles mais clairement fausses d'apres les passages ; "
        "5) la question se comprend seule et porte sur des connaissances civiques, pas sur les fiches. "
        "Reponds uniquement par un tableau JSON, sans texte autour, avec un element par question, "
        "dans le meme ordre, de la forme "
        '[{"passed": true, "reason": "..."}]. '
        "passed vaut true si la question est acceptee, sinon false. "
        "reason explique brievement le probleme quand passed vaut false."
    ),
    "critic_human": (
        "Theme : {theme}\n\n"
        "Verifie exactement {count} questions, dans l'ordre, chacune avec son groupe de passages.\n\n"
        "{items}"
    ),
}
