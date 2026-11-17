
import pandas as pd

from src.data.preprocessing import (
    clean_data,
    load_raw_data
)

from src.features.build_features import engineer_features


def test_cleaning_preserves_customer_count():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    assert len(cleaned) == len(raw)
    assert cleaned["customerID"].is_unique


def test_churn_encoding():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    assert set(cleaned["ChurnFlag"].unique()).issubset({0, 1})


def test_no_missing_total_charges():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    assert cleaned["TotalCharges"].notna().all()


def test_service_count_range():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    features = engineer_features(cleaned)

    assert features["service_count"].between(0, 8).all()


def test_feature_engineering_preserves_rows():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    features = engineer_features(cleaned)

    assert len(features) == len(cleaned)
    assert "tenure_group" in features.columns
    assert "is_month_to_month" in features.columns
