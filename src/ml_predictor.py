"""Leakage-safe baseline modelling for the Data Prism dashboard."""

import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
MIN_TRAINING_ROWS = 10
MAX_CATEGORICAL_LEVELS = 50


def predict_target(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Train a baseline model without fitting transforms on test data.

    The historical keys ``target_col``, ``metric`` and
    ``feature_importance_plot`` remain available for the dashboard. Additional
    structured fields make the evaluation auditable.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _error_report("Не удалось обучить модель: датасет пуст.")

    target_col = _select_target(df, target_column)
    if target_col is None:
        return _error_report("Не удалось выбрать целевую переменную.")

    modelling_data = df.dropna(subset=[target_col]).copy()
    if len(modelling_data) < MIN_TRAINING_ROWS:
        return _error_report(
            f"Недостаточно строк для оценки модели: нужно минимум {MIN_TRAINING_ROWS}.",
            target_col,
        )

    y = modelling_data[target_col]
    task_type = _infer_task_type(y)
    if task_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return _error_report("Целевая переменная содержит только один класс.", target_col)
        if int(class_counts.min()) < 3:
            return _error_report(
                "Для честного разделения и кросс-валидации нужно минимум 3 наблюдения в каждом классе.",
                target_col,
            )
    if task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_target = y.notna()
        modelling_data = modelling_data.loc[valid_target]
        y = y.loc[valid_target]
        if y.nunique(dropna=True) < 2:
            return _error_report("Числовая цель не содержит достаточной вариативности.", target_col)

    X, dropped_features = _prepare_features(modelling_data.drop(columns=[target_col]), y)
    if X.empty or not len(X.columns):
        return _error_report("После проверки качества не осталось пригодных признаков.", target_col)

    numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    try:
        X_train, X_test, y_train, y_test, split_notes = _split_data(X, y, task_type)
        model, model_comparison, cv_folds, cv_metric = _select_model_with_cv(
            X_train,
            y_train,
            task_type,
            numeric_columns,
            categorical_columns,
        )
        preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except ValueError as error:
        return _error_report(f"Не удалось сформировать корректную выборку: {error}", target_col)

    if task_type == "classification":
        metrics, metric_text, baseline_text = _classification_metrics(
            y_train, y_test, predictions
        )
    else:
        metrics, metric_text, baseline_text = _regression_metrics(y_train, y_test, predictions)

    diagnostics = _model_diagnostics(
        pipeline,
        X_test,
        y_test,
        predictions,
        task_type,
    )
    feature_importance_plot, top_features = _permutation_feature_importance(
        pipeline,
        X_test,
        y_test,
        task_type,
    )
    evaluation_notes = [
        split_notes,
        (
            f"Model family selected with {cv_folds}-fold cross-validation on the "
            f"training partition only ({cv_metric})."
        ),
        (
            "The holdout partition was reserved until model selection, then used for final metrics "
            "and permutation diagnostics."
        ),
        baseline_text,
        "Preprocessing was fitted on training rows only.",
        "Feature importance uses holdout permutation tests and is descriptive, not causal.",
    ]
    if task_type == "classification" and metrics["accuracy_lift"] <= 0:
        evaluation_notes.append(
            "Warning: the selected model did not beat the most-frequent holdout baseline on accuracy."
        )
    if task_type == "regression" and metrics["mae_improvement"] <= 0:
        evaluation_notes.append(
            "Warning: the selected model did not beat the mean holdout baseline on MAE."
        )
    if dropped_features:
        evaluation_notes.append(
            "Excluded potential leakage/identifier/no-signal features: "
            + ", ".join(dropped_features[:8])
        )

    return {
        "status": "ok",
        "target_col": str(target_col),
        "task_type": task_type,
        "model_name": type(model).__name__,
        "metric": metric_text,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "model_comparison": model_comparison,
        "cv_folds": cv_folds,
        "cv_metric": cv_metric,
        "feature_importance_plot": feature_importance_plot,
        "top_features": top_features,
        "feature_importance_method": "holdout permutation importance",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dropped_features": dropped_features,
        "evaluation_notes": evaluation_notes,
    }


def _select_target(df: pd.DataFrame, requested: Optional[str]) -> Optional[Any]:
    if requested and requested in df.columns:
        return requested
    candidates = [column for column in df.columns if df[column].nunique(dropna=True) >= 2]
    return candidates[-1] if candidates else None


def _infer_task_type(target: pd.Series) -> str:
    unique_count = int(target.nunique(dropna=True))
    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return "classification"
    classification_limit = max(20, int(math.sqrt(max(len(target), 1))))
    if pd.api.types.is_integer_dtype(target) and unique_count <= classification_limit:
        return "classification"
    return "regression"


def _prepare_features(X: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    prepared = X.copy()
    dropped: List[str] = []
    supported_columns = []

    for column in prepared.columns:
        series = prepared[column]
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / max(int(series.notna().sum()), 1)
        normalized_name = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name in {"index", "row_number"}
        ) and unique_ratio >= 0.95
        duplicates_target = series.reset_index(drop=True).equals(target.reset_index(drop=True))
        high_cardinality_text = (
            not pd.api.types.is_numeric_dtype(series)
            and unique_count > MAX_CATEGORICAL_LEVELS
            and unique_ratio >= 0.50
        )
        unsupported = pd.api.types.is_datetime64_any_dtype(series)

        if unique_count <= 1 or looks_like_id or duplicates_target or high_cardinality_text or unsupported:
            dropped.append(str(column))
        else:
            supported_columns.append(column)

    prepared = prepared[supported_columns]
    categorical_columns = prepared.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("object")
    return prepared, dropped


def _build_preprocessor(numeric_columns: List[Any], categorical_columns: List[Any]):
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=MAX_CATEGORICAL_LEVELS,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _candidate_models(task_type: str):
    if task_type == "classification":
        return [
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            RandomForestClassifier(
                n_estimators=120,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ]
    return [
        Ridge(alpha=1.0, solver="lsqr"),
        RandomForestRegressor(
            n_estimators=120,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ]


def _select_model_with_cv(
    X_train,
    y_train,
    task_type: str,
    numeric_columns: List[Any],
    categorical_columns: List[Any],
):
    if task_type == "classification":
        cv_folds = min(5, int(y_train.value_counts().min()))
        if cv_folds < 2:
            raise ValueError("Недостаточно примеров каждого класса в обучающей выборке.")
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        scoring = "balanced_accuracy"
        metric_label = "balanced accuracy; higher is better"
    else:
        cv_folds = min(5, len(X_train))
        if cv_folds < 2:
            raise ValueError("Недостаточно строк для кросс-валидации.")
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_mean_absolute_error"
        metric_label = "MAE; lower is better"

    comparison = []
    successful_candidates = []
    for model in _candidate_models(task_type):
        pipeline = Pipeline(
            [
                ("preprocessor", _build_preprocessor(numeric_columns, categorical_columns)),
                ("model", model),
            ]
        )
        model_name = type(model).__name__
        try:
            raw_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=splitter,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
        except (TypeError, ValueError) as error:
            comparison.append(
                {
                    "model_name": model_name,
                    "mean_score": None,
                    "std_score": None,
                    "status": "error",
                    "error": str(error),
                    "selected": False,
                }
            )
            continue

        raw_mean = float(np.mean(raw_scores))
        display_scores = raw_scores if task_type == "classification" else -raw_scores
        entry = {
            "model_name": model_name,
            "mean_score": round(float(np.mean(display_scores)), 6),
            "std_score": round(float(np.std(display_scores)), 6),
            "status": "ok",
            "error": None,
            "selected": False,
        }
        comparison.append(entry)
        successful_candidates.append((raw_mean, model, entry))

    if not successful_candidates:
        raise ValueError("Все модели завершили кросс-валидацию с ошибкой.")

    _, selected_model, selected_entry = max(successful_candidates, key=lambda item: item[0])
    selected_entry["selected"] = True
    return selected_model, comparison, cv_folds, metric_label


def _split_data(X, y, task_type: str):
    test_count = max(2, int(math.ceil(len(X) * 0.20)))
    test_count = min(test_count, len(X) - 2)
    stratify = None
    stratified = False
    if task_type == "classification":
        class_counts = y.value_counts()
        class_count = len(class_counts)
        maximum_test_count = len(X) - (2 * class_count)
        if maximum_test_count < class_count:
            raise ValueError(
                "Недостаточно строк, чтобы сохранить классы в train, test и кросс-валидации."
            )
        test_count = min(max(test_count, class_count), maximum_test_count)
        stratify = y
        stratified = True

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    split_name = "stratified holdout" if stratified else "holdout"
    note = f"Evaluation uses a deterministic {split_name}: {len(X_train)} train / {len(X_test)} test rows."
    return X_train, X_test, y_train, y_test, note


def _classification_metrics(y_train, y_test, predictions):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = baseline.predict(np.zeros((len(y_test), 1)))

    accuracy = float(accuracy_score(y_test, predictions))
    baseline_accuracy = float(accuracy_score(y_test, baseline_predictions))
    balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    metrics = {
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "weighted_f1": round(weighted_f1, 6),
        "baseline_accuracy": round(baseline_accuracy, 6),
        "accuracy_lift": round(accuracy - baseline_accuracy, 6),
    }
    metric_text = (
        f"Accuracy: {accuracy:.2%} (baseline: {baseline_accuracy:.2%}); "
        f"Balanced accuracy: {balanced_accuracy:.2%}; F1: {weighted_f1:.2%}"
    )
    baseline_text = f"Accuracy lift over most-frequent baseline: {accuracy - baseline_accuracy:+.2%}."
    return metrics, metric_text, baseline_text


def _regression_metrics(y_train, y_test, predictions):
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = baseline.predict(np.zeros((len(y_test), 1)))

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan")
    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
    improvement = 1 - (mae / baseline_mae) if baseline_mae > 0 else 0.0
    metrics = {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6) if np.isfinite(r2) else None,
        "baseline_mae": round(baseline_mae, 6),
        "mae_improvement": round(float(improvement), 6),
    }
    r2_text = f"{r2:.2%}" if np.isfinite(r2) else "N/A"
    metric_text = (
        f"MAE: {mae:,.3f} (baseline: {baseline_mae:,.3f}); "
        f"RMSE: {rmse:,.3f}; R²: {r2_text}"
    )
    baseline_text = f"MAE improvement over mean baseline: {improvement:+.2%}."
    return metrics, metric_text, baseline_text


def _model_diagnostics(pipeline, X_test, y_test, predictions, task_type: str):
    if task_type == "regression":
        residuals = np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)
        absolute_errors = np.abs(residuals)
        return {
            "type": "regression",
            "residual_summary": {
                "mean_error": round(float(np.mean(residuals)), 6),
                "residual_std": round(float(np.std(residuals)), 6),
                "median_absolute_error": round(float(np.median(absolute_errors)), 6),
                "p90_absolute_error": round(float(np.percentile(absolute_errors, 90)), 6),
                "max_absolute_error": round(float(np.max(absolute_errors)), 6),
            },
            "notes": [
                "Mean error near zero reduces evidence of aggregate bias but does not rule out subgroup bias.",
                "The 90th-percentile absolute error describes a typical high-error case on this holdout.",
            ],
        }

    model = pipeline.named_steps["model"]
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_class = [
        {
            "label": str(label),
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    probability_metrics = {}
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        confidence = np.max(probabilities, axis=1)
        correctness = np.asarray(predictions) == np.asarray(y_test)
        probability_metrics["expected_calibration_error"] = round(
            _expected_calibration_error(confidence, correctness), 6
        )
        try:
            if len(labels) == 2:
                binary_target = (np.asarray(y_test) == labels[1]).astype(int)
                probability_metrics["roc_auc"] = round(
                    float(roc_auc_score(binary_target, probabilities[:, 1])), 6
                )
                probability_metrics["brier_score"] = round(
                    float(brier_score_loss(binary_target, probabilities[:, 1])), 6
                )
            else:
                probability_metrics["roc_auc_ovr_weighted"] = round(
                    float(
                        roc_auc_score(
                            y_test,
                            probabilities,
                            labels=labels,
                            multi_class="ovr",
                            average="weighted",
                        )
                    ),
                    6,
                )
        except ValueError:
            probability_metrics["probability_note"] = (
                "Probability metrics require every evaluated class in the holdout."
            )

    return {
        "type": "classification",
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "values": matrix.astype(int).tolist(),
        },
        "per_class_metrics": per_class,
        "probability_metrics": probability_metrics,
        "notes": [
            "Per-class recall exposes failures hidden by aggregate accuracy.",
            "Calibration and discrimination are different; both should be reviewed before deployment.",
        ],
    }


def _expected_calibration_error(confidence, correctness, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(np.mean(correctness[mask])) - float(np.mean(confidence[mask]))
        )
    return error if total else 0.0


def _permutation_feature_importance(pipeline, X_test, y_test, task_type: str):
    scoring = "balanced_accuracy" if task_type == "classification" else "neg_mean_absolute_error"
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_names = [str(column) for column in X_test.columns]
    importances = np.asarray(result.importances_mean)
    order = np.argsort(importances)[::-1][:10]
    top_features = [
        {
            "feature": feature_names[index],
            "importance": round(float(importances[index]), 6),
            "importance_std": round(float(result.importances_std[index]), 6),
        }
        for index in order
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    display_items = list(reversed(top_features))
    ax.barh(
        [item["feature"] for item in display_items],
        [item["importance"] for item in display_items],
        xerr=[item["importance_std"] for item in display_items],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Holdout permutation importance")
    ax.set_xlabel("Decrease in evaluation score after permutation")
    fig.tight_layout()

    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    image.seek(0)
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", top_features


def _error_report(message: str, target_col: Optional[Any] = None) -> Dict[str, Any]:
    return {
        "status": "error",
        "target_col": str(target_col) if target_col is not None else "Не определена",
        "task_type": None,
        "model_name": None,
        "metric": f"⚠️ {message}",
        "metrics": {},
        "diagnostics": {},
        "model_comparison": [],
        "cv_folds": 0,
        "cv_metric": None,
        "feature_importance_plot": None,
        "top_features": [],
        "feature_importance_method": None,
        "train_rows": 0,
        "test_rows": 0,
        "dropped_features": [],
        "evaluation_notes": [message],
    }
