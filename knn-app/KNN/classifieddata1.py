"""Data and model helpers for the Classified Data KNN example."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
	"WTT",
	"PTI",
	"EQW",
	"SBI",
	"LQE",
	"QWG",
	"FDJ",
	"PJF",
	"HQE",
	"NXJ",
)
TARGET = "TARGET CLASS"
DEFAULT_NEIGHBORS = 23
DATA_PATH = Path(__file__).resolve().parents[1] / "Classified Data"


def load_dataset() -> pd.DataFrame:
	"""Read and validate the dataset stored next to the project files."""
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Could not find the dataset at {DATA_PATH}")

	dataset = pd.read_csv(DATA_PATH, index_col=0)
	required_columns = [*FEATURES, TARGET]
	missing_columns = [column for column in required_columns if column not in dataset]
	if missing_columns:
		raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
	if dataset[required_columns].isna().any().any():
		raise ValueError("Dataset contains missing values in required columns")

	labels = set(dataset[TARGET].unique())
	if not labels.issubset({0, 1}):
		raise ValueError("TARGET CLASS must contain only 0 and 1")
	return dataset


def get_dataset_summary() -> dict[str, Any]:
	dataset = load_dataset()
	counts = dataset[TARGET].value_counts()
	return {
		"row_count": len(dataset),
		"feature_count": len(FEATURES),
		"features": list(FEATURES),
		"class_counts": {label: int(counts.get(label, 0)) for label in (0, 1)},
	}


def _new_pipeline(neighbors: int = DEFAULT_NEIGHBORS) -> Pipeline:
	return Pipeline(
		[
			("scaler", StandardScaler()),
			("knn", KNeighborsClassifier(n_neighbors=neighbors)),
		]
	)


@lru_cache(maxsize=1)
def _fitted_prediction_model() -> Pipeline:
	dataset = load_dataset()
	model = _new_pipeline()
	model.fit(dataset[list(FEATURES)], dataset[TARGET])
	return model


def predict_one(values: dict[str, float]) -> dict[str, Any]:
	"""Predict one row; values must contain each feature in FEATURES order."""
	missing_features = [feature for feature in FEATURES if feature not in values]
	if missing_features:
		raise ValueError(f"Missing features: {', '.join(missing_features)}")

	row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
	model = _fitted_prediction_model()
	probabilities = model.predict_proba(row)[0]
	classes = model.named_steps["knn"].classes_
	return {
		"prediction": int(model.predict(row)[0]),
		"probabilities": {
			int(label): float(probability)
			for label, probability in zip(classes, probabilities)
		},
	}


@lru_cache(maxsize=1)
def evaluate_model() -> dict[str, Any]:
	"""Evaluate K=23 on held-out rows and compare K values using training-only CV."""
	dataset = load_dataset()
	features = dataset[list(FEATURES)]
	target = dataset[TARGET]
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.30,
		random_state=42,
		stratify=target,
	)

	model = _new_pipeline()
	model.fit(x_train, y_train)
	predictions = model.predict(x_test)

	cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	comparisons = []
	for neighbors in range(1, 40):
		scores = cross_val_score(
			_new_pipeline(neighbors),
			x_train,
			y_train,
			cv=cross_validation,
			scoring="accuracy",
		)
		comparisons.append({"neighbors": neighbors, "accuracy": float(scores.mean())})

	best = max(comparisons, key=lambda item: item["accuracy"])
	matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
	return {
		"accuracy": float(accuracy_score(y_test, predictions)),
		"confusion_matrix": matrix.tolist(),
		"test_rows": len(y_test),
		"neighbors": DEFAULT_NEIGHBORS,
		"cv_comparisons": comparisons,
		"best_cv_neighbors": int(best["neighbors"]),
		"best_cv_accuracy": float(best["accuracy"]),
	}
