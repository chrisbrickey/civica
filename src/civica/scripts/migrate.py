import logging

from dotenv import load_dotenv

import civica.db.migrate as db_migrate

logger = logging.getLogger(__name__)


def main() -> None:
    """Apply database schema and confirm."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    db_migrate.apply_schema()
    logger.info("Schema applied.")


if __name__ == "__main__":
    main()
