"""Deterministic anomaly review ranking and guarded exploratory segmentation."""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from src.column_semantics import columns_for_role, profile_dataframe_columns


RANDOM_STATE = 42
MIN_ANALYSIS_ROWS = 20
MAX_FEATURES = 12
MAX_SEGMENTS = 6
ANOMALY_CONTAMINATION = 0.03
GAP_REFERENCE_REPEATS = 5
STABILITY_SUBSAMPLE_SHARE = 0.80
STABILITY_SEEDS = (43, 44, 45, 46, 47)


class AnomalySegmentationEngine:
    """Find unusual rows and exploratory segments using robust numeric features."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        column_profiles: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df
        self.column_profiles = (
            dict(column_profiles)
            if column_profiles is not None
            else profile_dataframe_columns(df)
        )

    def analyze(self, max_anomalies: int = 10) -> Dict[str, Any]:
        features = self._feature_columns()
        if len(self.df) < MIN_ANALYSIS_ROWS or not features:
            return self._insufficient_report(features)

        values, scaled = self._prepare(features)
        anomaly_report, anomaly_labels = self._detect_anomalies(
            features,
            values,
            scaled,
            max_anomalies,
        )
        segmentation_report = self._segment(features, values, scaled, anomaly_labels)
        return {
            "status": "ok",
            "row_count": int(len(self.df)),
            "features": [str(feature) for feature in features],
            "anomaly_detection": anomaly_report,
            "segmentation": segmentation_report,
            "methodology_notes": [
                "Isolation Forest ranks a fixed review budget of multivariate candidates; flags are not an estimate of anomaly prevalence or proof of bad data or fraud.",
                "Median imputation and robust scaling are fitted on the analyzed dataset.",
                "The gap statistic must support more than one cluster before K-means segments are reported.",
                "Segmentation stability is measured by refitting K-means on deterministic 80% subsamples and comparing full-data assignments.",
                "K-means segmentation excludes review candidates while fitting centroids, then assigns every row.",
                "Segments are exploratory and require domain validation before operational use.",
            ],
        }

    def evidence_insights(self, report: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        report = report or self.analyze()
        if report.get("status") != "ok":
            return []

        anomaly = report["anomaly_detection"]
        segmentation = report["segmentation"]
        insights = [
            {
                "insight_id": "multivariate-anomaly-scan",
                "category": "anomaly",
                "title": "Multivariate anomaly review candidates",
                "statement": anomaly["summary"],
                "evidence": [
                    f"Flagged rows: {anomaly['flagged_count']} of {anomaly['evaluated_rows']}",
                    f"Flagged rate: {anomaly['flagged_rate']:.2%}",
                    f"Features used: {len(report['features'])}",
                ],
                "recommendation": "Review high-scoring rows with domain owners; do not delete them automatically.",
                "confidence": "exploratory",
                "severity": "info",
                "sample_size": anomaly["evaluated_rows"],
                "metrics": {
                    "flagged_count": anomaly["flagged_count"],
                    "flagged_rate": anomaly["flagged_rate"],
                },
                "type": "evidence",
            }
        ]
        if segmentation.get("status") == "ok" and segmentation.get("quality") in {
            "moderate",
            "strong",
        }:
            insights.append(
                {
                    "insight_id": "exploratory-segmentation",
                    "category": "segmentation",
                    "title": "Exploratory segment structure",
                    "statement": segmentation["summary"],
                    "evidence": [
                        f"Selected segments: {segmentation['selected_k']}",
                        f"Silhouette score: {segmentation['silhouette_score']:.3f}",
                        f"Stability (adjusted Rand index): {segmentation['stability_score']:.3f}",
                    ],
                    "recommendation": (
                        "Validate segment usefulness against business outcomes and future data before activation."
                    ),
                    "confidence": segmentation["quality"],
                    "severity": "info" if segmentation["quality"] != "weak" else "warning",
                    "sample_size": segmentation["fit_rows"],
                    "metrics": {
                        "selected_k": segmentation["selected_k"],
                        "silhouette_score": segmentation["silhouette_score"],
                        "stability_score": segmentation["stability_score"],
                    },
                    "type": "evidence",
                }
            )
        return insights

    def _feature_columns(self) -> List[Any]:
        selected = []
        for column in columns_for_role(
            self.df,
            "measure",
            profiles=self.column_profiles,
            require_variation=True,
        ):
            if (
                not pd.api.types.is_numeric_dtype(self.df[column])
                or pd.api.types.is_bool_dtype(self.df[column])
                or pd.api.types.is_complex_dtype(self.df[column])
            ):
                continue
            series = pd.to_numeric(self.df[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if series.nunique(dropna=True) < 2:
                continue
            selected.append(column)
        return selected[:MAX_FEATURES]

    def _prepare(self, features: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.df[features].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        values = SimpleImputer(strategy="median").fit_transform(frame)
        scaled = RobustScaler().fit_transform(values)
        return np.asarray(values, dtype=float), np.asarray(scaled, dtype=float)

    def _detect_anomalies(
        self,
        features: List[Any],
        values: np.ndarray,
        scaled: np.ndarray,
        max_anomalies: int,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        detector = IsolationForest(
            n_estimators=200,
            contamination=ANOMALY_CONTAMINATION,
            max_samples=min(256, len(scaled)),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        labels = detector.fit_predict(scaled)
        raw_scores = -detector.decision_function(scaled)
        score_range = float(np.ptp(raw_scores))
        scores = (
            (raw_scores - float(np.min(raw_scores))) / score_range
            if score_range > 0
            else np.zeros(len(raw_scores))
        )
        flagged_positions = np.flatnonzero(labels == -1)
        ranked_positions = flagged_positions[np.argsort(scores[flagged_positions])[::-1]]
        centers = np.median(values, axis=0)
        scales = _robust_feature_scales(values)
        top_anomalies = []
        for position in ranked_positions[:max(0, max_anomalies)]:
            deviations = (values[position] - centers) / scales
            reason_order = np.argsort(np.abs(deviations))[::-1][:3]
            reasons = [
                {
                    "feature": str(features[index]),
                    "value": round(float(values[position, index]), 6),
                    "median": round(float(centers[index]), 6),
                    "robust_deviation": round(float(deviations[index]), 3),
                }
                for index in reason_order
            ]
            top_anomalies.append(
                {
                    "row_position": int(position),
                    "row_index": str(self.df.index[position]),
                    "anomaly_score": round(float(scores[position]), 6),
                    "reasons": reasons,
                }
            )

        flagged_count = int(len(flagged_positions))
        flagged_rate = flagged_count / len(scaled)
        return (
            {
                "status": "ok",
                "method": "Isolation Forest",
                "evaluated_rows": int(len(scaled)),
                "expected_contamination": ANOMALY_CONTAMINATION,
                "candidate_policy": "fixed_review_budget",
                "review_budget_rate": ANOMALY_CONTAMINATION,
                "flagged_count": flagged_count,
                "flagged_rate": round(float(flagged_rate), 6),
                "summary": (
                    f"Isolation Forest ranked {flagged_count} of {len(scaled)} rows "
                    f"using a configured {ANOMALY_CONTAMINATION:.1%} review budget; "
                    "these are candidates, not confirmed anomalies."
                ),
                "top_anomalies": top_anomalies,
            },
            labels,
        )

    def _segment(
        self,
        features: List[Any],
        values: np.ndarray,
        scaled: np.ndarray,
        anomaly_labels: np.ndarray,
    ) -> Dict[str, Any]:
        if len(features) < 2:
            return {
                "status": "insufficient_data",
                "method": "Gap statistic gate followed by K-means",
                "selected_k": None,
                "gap_selected_k": None,
                "silhouette_score": None,
                "stability_score": None,
                "quality": "insufficient",
                "fit_rows": int(len(scaled)),
                "excluded_anomaly_rows": 0,
                "candidate_scores": [],
                "gap_scores": [],
                "summary": (
                    "Segmentation requires at least two varying numeric features."
                ),
                "segments": [],
            }

        fit_mask = anomaly_labels != -1
        fit_scaled = scaled[fit_mask]
        if len(fit_scaled) < MIN_ANALYSIS_ROWS:
            fit_mask = np.ones(len(scaled), dtype=bool)
            fit_scaled = scaled

        unique_fit_rows = int(np.unique(fit_scaled, axis=0).shape[0])
        max_k = min(MAX_SEGMENTS, len(fit_scaled) - 1, unique_fit_rows)
        if max_k < 2:
            return {
                "status": "insufficient_evidence",
                "method": "Gap statistic gate followed by K-means",
                "selected_k": None,
                "gap_selected_k": 1,
                "silhouette_score": None,
                "stability_score": None,
                "quality": "insufficient",
                "fit_rows": int(len(fit_scaled)),
                "excluded_anomaly_rows": int((~fit_mask).sum()),
                "candidate_scores": [],
                "gap_scores": [],
                "summary": (
                    "Fewer than two distinct multivariate patterns remain after preparation."
                ),
                "segments": [],
            }
        candidates = []
        fitted_models = {}
        for cluster_count in range(2, max_k + 1):
            model = KMeans(
                n_clusters=cluster_count,
                random_state=RANDOM_STATE,
                n_init=10,
            )
            labels = model.fit_predict(fit_scaled)
            counts = np.bincount(labels, minlength=cluster_count)
            if np.any(counts < 2):
                continue
            score = float(
                silhouette_score(
                    fit_scaled,
                    labels,
                    sample_size=min(2000, len(fit_scaled)),
                    random_state=RANDOM_STATE,
                )
            )
            candidates.append(
                {
                    "k": cluster_count,
                    "silhouette_score": round(score, 6),
                    "smallest_segment_share": round(float(counts.min() / len(labels)), 6),
                }
            )
            fitted_models[cluster_count] = model

        if not candidates:
            return {
                "status": "insufficient_data",
                "method": "Gap statistic gate followed by K-means",
                "selected_k": None,
                "gap_selected_k": None,
                "silhouette_score": None,
                "stability_score": None,
                "quality": "insufficient",
                "fit_rows": int(len(fit_scaled)),
                "excluded_anomaly_rows": int((~fit_mask).sum()),
                "candidate_scores": [],
                "gap_scores": [],
                "summary": "No valid segmentation could be fitted.",
                "segments": [],
            }

        gap_selected_k, gap_scores = self._gap_statistic(fit_scaled, max_k)
        if gap_selected_k == 1:
            return {
                "status": "insufficient_evidence",
                "method": "Gap statistic gate followed by K-means",
                "selected_k": None,
                "gap_selected_k": 1,
                "silhouette_score": None,
                "stability_score": None,
                "quality": "insufficient",
                "fit_rows": int(len(fit_scaled)),
                "excluded_anomaly_rows": int((~fit_mask).sum()),
                "candidate_scores": candidates,
                "gap_scores": gap_scores,
                "segments": [],
                "summary": (
                    "The gap statistic selected one cluster, so the data does not "
                    "provide sufficient evidence for a multi-segment solution."
                ),
            }

        balanced_candidates = [
            item
            for item in candidates
            if item["smallest_segment_share"] >= 0.05
            and item["k"] <= gap_selected_k
        ]
        gap_candidates = [item for item in candidates if item["k"] <= gap_selected_k]
        if not gap_candidates:
            return {
                "status": "insufficient_evidence",
                "method": "Gap statistic gate followed by K-means",
                "selected_k": None,
                "gap_selected_k": gap_selected_k,
                "silhouette_score": None,
                "stability_score": None,
                "quality": "insufficient",
                "fit_rows": int(len(fit_scaled)),
                "excluded_anomaly_rows": int((~fit_mask).sum()),
                "candidate_scores": candidates,
                "gap_scores": gap_scores,
                "segments": [],
                "summary": (
                    "The gap statistic supported multiple clusters, but no partition "
                    "within that bound retained at least two rows per segment."
                ),
            }
        selection_pool = balanced_candidates or gap_candidates
        selected = max(selection_pool, key=lambda item: (item["silhouette_score"], -item["k"]))
        selected_k = int(selected["k"])
        model = fitted_models[selected_k]
        all_labels = model.predict(scaled)
        all_labels = _stable_segment_labels(all_labels)
        fit_labels = model.predict(fit_scaled)
        stability = self._cluster_stability(fit_scaled, fit_labels, selected_k)
        quality = _segmentation_quality(
            selected["silhouette_score"],
            stability,
            selected["smallest_segment_share"],
        )
        segments = self._segment_profiles(features, values, all_labels, selected_k)
        return {
            "status": "ok",
            "method": "Gap-gated K-means selected by silhouette score",
            "selected_k": selected_k,
            "gap_selected_k": gap_selected_k,
            "silhouette_score": selected["silhouette_score"],
            "stability_score": round(stability, 6),
            "quality": quality,
            "fit_rows": int(len(fit_scaled)),
            "excluded_anomaly_rows": int((~fit_mask).sum()),
            "candidate_scores": candidates,
            "gap_scores": gap_scores,
            "segments": segments,
            "summary": (
                f"The best exploratory solution contains {selected_k} segments with "
                f"{quality} support (silhouette={selected['silhouette_score']:.3f}, "
                f"stability={stability:.3f})."
            ),
        }

    def _gap_statistic(
        self,
        values: np.ndarray,
        max_k: int,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Return the gap-statistic clusterability decision and diagnostics.

        The one-standard-error rule may select ``k=1``. In that case the
        engine must not manufacture a multi-segment result merely because
        K-means can partition any finite dataset.
        """
        lower = np.min(values, axis=0)
        spans = np.maximum(np.max(values, axis=0) - lower, 1e-12)
        rng = np.random.default_rng(RANDOM_STATE)
        references = [
            lower + rng.random(values.shape) * spans
            for _ in range(GAP_REFERENCE_REPEATS)
        ]
        diagnostics = []
        raw_gaps = []
        raw_standard_errors = []
        for cluster_count in range(1, max_k + 1):
            observed = KMeans(
                n_clusters=cluster_count,
                random_state=RANDOM_STATE,
                n_init=10,
            ).fit(values)
            reference_logs = []
            for reference_index, reference in enumerate(references):
                reference_model = KMeans(
                    n_clusters=cluster_count,
                    random_state=RANDOM_STATE + reference_index + 1,
                    n_init=10,
                ).fit(reference)
                reference_logs.append(
                    float(np.log(max(float(reference_model.inertia_), 1e-12)))
                )
            reference_values = np.asarray(reference_logs, dtype=float)
            gap = float(
                np.mean(reference_values)
                - np.log(max(float(observed.inertia_), 1e-12))
            )
            standard_error = float(
                np.std(reference_values, ddof=1)
                * np.sqrt(1.0 + 1.0 / GAP_REFERENCE_REPEATS)
            )
            raw_gaps.append(gap)
            raw_standard_errors.append(standard_error)
            diagnostics.append(
                {
                    "k": cluster_count,
                    "gap": round(gap, 6),
                    "standard_error": round(standard_error, 6),
                }
            )

        selected_k = int(diagnostics[-1]["k"])
        for index in range(len(diagnostics) - 1):
            if raw_gaps[index] >= raw_gaps[index + 1] - raw_standard_errors[index + 1]:
                selected_k = int(diagnostics[index]["k"])
                break
        return selected_k, diagnostics

    def _cluster_stability(self, values: np.ndarray, reference: np.ndarray, k: int) -> float:
        """Measure assignment stability after deterministic row subsampling."""
        scores = []
        sample_size = max(k * 2, int(np.ceil(len(values) * STABILITY_SUBSAMPLE_SHARE)))
        sample_size = min(sample_size, len(values))
        for seed in STABILITY_SEEDS:
            rng = np.random.default_rng(seed)
            positions = np.sort(
                rng.choice(len(values), size=sample_size, replace=False)
            )
            model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(
                values[positions]
            )
            labels = model.predict(values)
            scores.append(float(adjusted_rand_score(reference, labels)))
        return float(np.mean(scores))

    def _segment_profiles(
        self,
        features: List[Any],
        values: np.ndarray,
        labels: np.ndarray,
        segment_count: int,
    ) -> List[Dict[str, Any]]:
        overall_means = np.mean(values, axis=0)
        scales = _robust_feature_scales(values)
        profiles = []
        for label in range(segment_count):
            mask = labels == label
            segment_means = np.mean(values[mask], axis=0)
            differences = (segment_means - overall_means) / scales
            top = np.argsort(np.abs(differences))[::-1][:3]
            profiles.append(
                {
                    "segment": int(label + 1),
                    "size": int(mask.sum()),
                    "share": round(float(mask.mean()), 6),
                    "distinguishing_features": [
                        {
                            "feature": str(features[index]),
                            "segment_mean": round(float(segment_means[index]), 6),
                            "overall_mean": round(float(overall_means[index]), 6),
                            "robust_difference": round(float(differences[index]), 3),
                        }
                        for index in top
                    ],
                }
            )
        return profiles

    def _insufficient_report(self, features: List[Any]) -> Dict[str, Any]:
        return {
            "status": "insufficient_data",
            "row_count": int(len(self.df)),
            "features": [str(feature) for feature in features],
            "summary": (
                f"At least {MIN_ANALYSIS_ROWS} rows and one varying numeric feature are required."
            ),
            "anomaly_detection": {"status": "insufficient_data", "top_anomalies": []},
            "segmentation": {"status": "insufficient_data", "segments": []},
            "methodology_notes": [],
        }


def _robust_feature_scales(values: np.ndarray) -> np.ndarray:
    lower = np.percentile(values, 25, axis=0)
    upper = np.percentile(values, 75, axis=0)
    scales = (upper - lower) / 1.349
    standard_deviations = np.std(values, axis=0)
    scales = np.where(scales > 1e-12, scales, standard_deviations)
    return np.where(scales > 1e-12, scales, 1.0)


def _stable_segment_labels(labels: np.ndarray) -> np.ndarray:
    counts = [(int(label), int(np.sum(labels == label))) for label in np.unique(labels)]
    ordered = sorted(counts, key=lambda item: (-item[1], item[0]))
    mapping = {old: new for new, (old, _) in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _segmentation_quality(
    silhouette: float,
    stability: float,
    smallest_segment_share: float,
) -> str:
    if smallest_segment_share < 0.05:
        return "weak"
    if silhouette >= 0.50 and stability >= 0.80:
        return "strong"
    if silhouette >= 0.25 and stability >= 0.50:
        return "moderate"
    return "weak"
