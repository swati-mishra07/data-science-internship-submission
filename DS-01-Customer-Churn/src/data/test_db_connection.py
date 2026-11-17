
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load database credentials
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Check your .env file.")

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        version = connection.execute(
            text("SELECT version();")
        ).scalar()

        database = connection.execute(
            text("SELECT current_database();")
        ).scalar()

        print("PostgreSQL connection successful!")
        print(f"Database: {database}")
        print(f"Version: {version}")

except Exception as error:
    print("PostgreSQL connection failed.")
    print(error)

finally:
    engine.dispose()
