
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_cleaned.csv"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_features.csv"
)

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL)


def load_csv_to_postgres(file_path: Path, table_name: str):
    """Load a processed CSV dataset into PostgreSQL."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}. "
            "Run preprocessing and feature engineering first."
        )

    df = pd.read_csv(file_path)

    print(f"\nLoading table: {table_name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=1000
    )

    print(f"Successfully loaded {table_name}")


def verify_tables():
    """Check whether the PostgreSQL tables contain records."""

    table_names = ["telco_cleaned", "telco_features"]

    with engine.connect() as connection:
        for table_name in table_names:
            result = connection.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            )

            print(f"{table_name}: {result.scalar()} records")


if __name__ == "__main__":
    try:
        load_csv_to_postgres(
            CLEAN_DATA_PATH,
            "telco_cleaned"
        )

        load_csv_to_postgres(
            FEATURE_DATA_PATH,
            "telco_features"
        )

        print("\nVerifying PostgreSQL tables...")
        verify_tables()

        print("\nDatabase ingestion completed successfully!")

    finally:
        engine.dispose()
