from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.db import get_connection, init_db


def main() -> None:
    conn = get_connection()
    init_db(conn)
    print("Database initialized.")


if __name__ == "__main__":
    main()
