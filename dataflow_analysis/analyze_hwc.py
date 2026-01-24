#!/usr/bin/env python3

import argparse
import pathlib
import sqlite3
import pandas as pd


def read_sqlite_db(db_path: pathlib.Path):
    print(f"\n=== Database: {db_path} ===")

    conn = sqlite3.connect(db_path)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )["name"].tolist()

        if not tables:
            print("  (No tables found)")
            return

        for table in tables:
            print(f"\n--- Table: {table} ---")
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            print(df)

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Read SQLite database files into pandas and print them"
    )
    parser.add_argument(
        "path",
        type=pathlib.Path,
        default=pathlib.Path("/scratch/ybudanaz/metrics2/npbench_papi_intel_xeon_6154/L/npbench_papi_metrics.db"),
        help="Path to a SQLite file or a directory containing SQLite files",
    )
    parser.add_argument(
        "--pattern",
        default="*.db",
        help="Filename pattern to match (default: *.db)",
    )

    args = parser.parse_args()

    if args.path.is_file():
        read_sqlite_db(args.path)
    elif args.path.is_dir():
        db_files = sorted(args.path.glob(args.pattern))
        if not db_files:
            print(f"No database files matching '{args.pattern}' found in {args.path}")
            return

        for db in db_files:
            read_sqlite_db(db)
    else:
        raise FileNotFoundError(f"{args.path} does not exist")


if __name__ == "__main__":
    main()
