# test_app.py

import pytest
from fastapi.testclient import TestClient
from main import app, NLIRequest  # Adjust the import according to your file structure

client = TestClient(app)

def test_predict_with_hypothesis():
    response = client.post(
        "/predict",
        json={"premise": "The cat is on the mat.", "hypothesis": "The mat is under the cat."}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "label" in json_response
    assert "scores" in json_response
    assert "labels" in json_response

def test_predict_with_labels():
    response = client.post(
        "/predict",
        json={"premise": "The cat is on the mat.", "labels": ["entailment", "contradiction", "neutral"]}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "label" in json_response
    assert "scores" in json_response
    assert "labels" in json_response

def test_predict_missing_premise():
    response = client.post(
        "/predict",
        json={"hypothesis": "The mat is under the cat."}
    )
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_predict_missing_hypothesis_and_labels():
    response = client.post(
        "/predict",
        json={"premise": "The cat is on the mat."}
    )
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_predict_invalid_json():
    response = client.post(
        "/predict",
        json={"premise": 12345}  # Invalid type for premise
    )
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

