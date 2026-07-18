from dotenv import load_dotenv

import civica.db.migrate as db_migrate


def main() -> None:
    """Apply database schema and confirm."""
    load_dotenv()
    db_migrate.apply_schema()
    print("Schema applied.")


if __name__ == "__main__":
    main()
