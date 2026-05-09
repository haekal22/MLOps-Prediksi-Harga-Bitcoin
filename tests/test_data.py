import pandas as pd

def test_dataset_not_empty():
    df = pd.read_csv("data/processed/btc_features.csv")

    assert len(df) > 0


def test_target_exists():
    df = pd.read_csv("data/processed/btc_features.csv")

    assert "target" in df.columns