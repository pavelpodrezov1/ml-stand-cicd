from pathlib import Path


def test_model_artifact_exists() -> None:
    model_path = Path("model") / "titanic_model.joblib"
    assert model_path.exists(), f"Model file not found: {model_path}"
