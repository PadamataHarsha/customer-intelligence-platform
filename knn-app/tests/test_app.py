from fastapi.testclient import TestClient

from KNN.classifieddata1 import FEATURES, evaluate_model, get_dataset_summary, predict_one
from main import app


client = TestClient(app)


def test_dataset_has_expected_schema_and_two_classes():
    summary = get_dataset_summary()

    assert summary["feature_count"] == 10
    assert summary["features"] == list(FEATURES)
    assert set(summary["class_counts"]) == {0, 1}
    assert sum(summary["class_counts"].values()) == summary["row_count"]


def test_prediction_returns_class_and_probabilities():
    result = predict_one({feature: 1.0 for feature in FEATURES})

    assert result["prediction"] in {0, 1}
    assert set(result["probabilities"]) == {0, 1}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9


def test_evaluation_is_reproducible_and_includes_k_comparison():
    first = evaluate_model()
    second = evaluate_model()

    assert first == second
    assert len(first["cv_comparisons"]) == 39
    assert first["neighbors"] == 23
    assert len(first["confusion_matrix"]) == 2


def test_pages_and_health_route_load():
    for path in ("/", "/predict", "/evaluation"):
        response = client.get(path)
        assert response.status_code == 200
        assert "KNN" in response.text

    assert client.get("/health").json() == {"status": "ok"}


def test_prediction_form_reports_invalid_values():
    response = client.post("/predict", data={"WTT": "not-a-number"})

    assert response.status_code == 422
    assert "Enter a finite number." in response.text


def test_prediction_form_renders_a_successful_result():
    response = client.post("/predict", data={feature: "1.0" for feature in FEATURES})

    assert response.status_code == 200
    assert "PREDICTION RESULT" in response.text
    assert "Nearest-neighbor vote" in response.text


def test_json_prediction_api_validates_fields():
    payload = {feature: 1.0 for feature in FEATURES}
    valid_response = client.post("/api/predict", json=payload)
    invalid_response = client.post("/api/predict", json={"WTT": 1.0})

    assert valid_response.status_code == 200
    assert valid_response.json()["prediction"] in {0, 1}
    assert invalid_response.status_code == 422