import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predict import predict_yield


def test_predict_returns_float():
    result = predict_yield(
        22.0,
        88.0,
        920.0
    )

    assert isinstance(result, float)


def test_predict_non_negative():
    result = predict_yield(
        22.0,
        88.0,
        920.0
    )

    assert result >= 0