import argparse
import csv
import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Define settings for the CSV cleaning process
INPUT_CSV = Path(__file__).resolve().with_name("merged.csv")
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S %z"
TIMESTAMP_COLUMN = "timeinstant"

# Main cleaning function
def clean_csv(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output path must be different from the input path.")

    row_count = 0

    # Create a temporary SQLite database to store unique rows by timestamp (this is done for performance reasons)
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "timestamps.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE rows_by_timestamp (
                    timestamp_utc TEXT PRIMARY KEY,
                    row_data TEXT NOT NULL
                )
                """
            )

            # Read the input CSV file and process each row
            with input_path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source)
                try:
                    header = next(reader)
                except StopIteration as error:
                    raise ValueError(f"{input_path} is empty.") from error

                if header.count(TIMESTAMP_COLUMN) != 1:
                    raise ValueError(
                        f"Expected exactly one {TIMESTAMP_COLUMN!r} column in {input_path}."
                    )

                timestamp_index = header.index(TIMESTAMP_COLUMN)
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(
                            f"Expected {len(header)} columns at CSV record "
                            f"{reader.line_num}, got {len(row)}."
                        )

                    try:
                        timestamp = datetime.strptime(
                            row[timestamp_index], TIMESTAMP_FORMAT
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid timestamp at CSV record {reader.line_num}: "
                            f"{row[timestamp_index]!r}"
                        ) from error

                    timestamp_utc = timestamp.astimezone(timezone.utc).isoformat(
                        timespec="microseconds"
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO rows_by_timestamp
                            (timestamp_utc, row_data)
                        VALUES (?, ?)
                        """,
                        (timestamp_utc, json.dumps(row, ensure_ascii=False)),
                    )
                    row_count += 1

            connection.commit()
            unique_count = connection.execute(
                "SELECT COUNT(*) FROM rows_by_timestamp"
            ).fetchone()[0]

            # Write the cleaned and sorted data to the output CSV file
            with output_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(header)
                for (row_data,) in connection.execute(
                    "SELECT row_data FROM rows_by_timestamp ORDER BY timestamp_utc"
                ):
                    writer.writerow(json.loads(row_data))

    # Print summary of the cleaning process
    print(f"Input rows: {row_count}")
    print(f"Unique timestamps retained: {unique_count}")
    print(f"Duplicate rows removed: {row_count - unique_count}")
    print(f"Cleaned file: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate timestamps and sort a CSV chronologically."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=INPUT_CSV,
        help=f"input CSV path (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV path (default: <input name>_cleaned.csv)",
    )
    arguments = parser.parse_args()
    output_path = arguments.output or arguments.input.with_name(
        f"{arguments.input.stem}_cleaned{arguments.input.suffix}"
    )
    print(f"Cleaning {arguments.input} -> {output_path}")
    clean_csv(arguments.input, output_path)

if __name__ == "__main__":
    main()