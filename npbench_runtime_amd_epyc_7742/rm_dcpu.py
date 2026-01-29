#!/usr/bin/env python3
import sqlite3
import argparse

def delete_dace_cpu_rows(db_path, table_name, dry_run=False):
    """
    Delete all rows where framework column is 'dace_cpu'.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to delete from
        dry_run: If True, only count rows without deleting
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Count rows to be deleted
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE framework = ?",
            ("dace_cpu",)
        )
        count = cursor.fetchone()[0]
        
        print(f"Found {count} rows with framework='dace_cpu' in table '{table_name}'")
        
        if dry_run:
            print("DRY RUN: No rows deleted.")
        else:
            # Delete the rows
            cursor.execute(
                f"DELETE FROM {table_name} WHERE framework = ?",
                ("dace_cpu",)
            )
            conn.commit()
            print(f"Deleted {count} rows.")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
    
    finally:
        conn.close()


def main():
    delete_dace_cpu_rows("L/npbench.db", "results", False)


if __name__ == "__main__":
    main()