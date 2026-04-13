from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict() -> None:
    response = client.post(
        "/predict",
        json={
            "pclass": 3,
            "sex": "male",
            "age": 22,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "S",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "label" in payload
    assert payload["prediction"] in (0, 1)
