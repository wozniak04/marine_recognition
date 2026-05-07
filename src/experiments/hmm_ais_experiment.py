from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ais_preprocessing import AISPreprocessor
from src.data.dataset import load_ais_csv, load_port_database
from src.data.outliers import detect_teleports, fix_teleports
from src.data.ports import build_port_matcher, discover_ports
from src.experiments.registry import register_experiment
from src.models.hmm import HMM_TO_STATE, ShipHMM
from src.models.rule_engine import ShipStateFSM
from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)

GAP_THRESHOLD_S = 1800

STATE_COLUMNS = {
    "port_stay": "In Port",
    "voyage": "At Sea Voyage",
    "anchor": "At Sea Anchor",
    "adrift": "At Sea Adrift",
}


@register_experiment("hmm_ais")
class HMMAISExperiment:

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = create_logger(config.experiment_name)
        _, _, self.reports_dir = config.experiment_paths()

    @classmethod
    def name(cls) -> str:
        return "hmm_ais"

    def run(self) -> dict:
        ais_file = self.config.model.params.get(
            "ais_file", "Ship_Operation_example_dateset_AIS_based_data.csv"
        )
        ais_path = Path(self.config.paths.raw_dir).parent / "unclassified" / ais_file

        ais_df = load_ais_csv(ais_path)
        preprocessor = AISPreprocessor()
        ais_df = preprocessor.process(ais_df)

        teleport_mask = detect_teleports(ais_df, max_speed_knots=self.config.outlier.max_speed_knots)
        if teleport_mask.any():
            ais_df = fix_teleports(ais_df, teleport_mask)

        ports_df = load_port_database(self.config)
        discovered = discover_ports(ais_df)
        matcher = build_port_matcher(ports_df, discovered, self.config.port)
        port_info = matcher.query_batch(ais_df["LAT"].values, ais_df["LON"].values)
        ais_df = ais_df.join(port_info)

        hmm = ShipHMM(self.config.model.params)
        gps_model_path = self._find_gps_model()

        if gps_model_path is not None:
            hmm.load(gps_model_path)
            self.logger.info("Loaded pre-trained HMM from %s", gps_model_path)
        else:
            self.logger.warning("No pre-trained HMM found — training unsupervised on AIS data")
            ais_df = self._compute_features(ais_df)
            feature_cols = hmm.feature_cols
            feature_cols = [c for c in feature_cols if c in ais_df.columns]
            X = ais_df[feature_cols].fillna(0).values
            hmm.model = self._fit_unsupervised(X, hmm)

        ais_df = self._compute_features(ais_df)
        classified = self._predict(hmm, ais_df)

        self._log_state_distribution(classified)
        classified.to_csv(self.reports_dir / "ais_hmm_classified.csv", index=False)
        self._export_csv(classified)

        comparison = self._compare_with_gps(classified)

        results = {
            "ais_rows": len(classified),
            "state_distribution": classified["predicted_state"].value_counts().to_dict(),
            "comparison": comparison,
        }

        self.logger.info("HMM AIS experiment complete")
        return results

    def _find_gps_model(self) -> Path | None:
        for exp_name in ["hmm_gps_v1", "hmm_gps_v2"]:
            model_path = Path("models") / exp_name
            if (model_path / "hmm_model.joblib").exists():
                return model_path
        return None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        time = pd.to_datetime(df["signaldate"])
        dt = time.diff().dt.total_seconds().values
        is_gap = np.zeros(len(df), dtype=bool)
        is_gap[0] = True
        is_gap[1:] = np.nan_to_num(dt[1:], nan=9999) > GAP_THRESHOLD_S
        df["is_gap"] = is_gap.astype(int)

        fsm = ShipStateFSM(self.config.model.params)
        gap_boundaries = fsm._find_gap_boundaries(df)
        df = fsm._compute_rolling_features(df, gap_boundaries)
        return df

    def _predict(self, hmm: ShipHMM, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = hmm.feature_cols
        feature_cols = [c for c in feature_cols if c in df.columns]

        gap_col = df["is_gap"].values if "is_gap" in df.columns else np.zeros(len(df))
        gap_indices = np.where(gap_col == 1)[0]
        boundaries = sorted(set(gap_indices.tolist()) | {0})

        segments = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(df)
            if end > start:
                segments.append((start, end))

        all_X = []
        lengths = []
        for start, end in segments:
            seg = df.iloc[start:end]
            all_X.append(seg[feature_cols].fillna(0).values)
            lengths.append(end - start)

        X = np.vstack(all_X)

        in_port = df["in_port"].values if "in_port" in df.columns else None
        state_indices, confidences = hmm.predict(X, lengths, in_port=in_port)

        result = df.copy()
        result["predicted_state"] = [HMM_TO_STATE[i] for i in state_indices]
        result["confidence"] = confidences
        return result

    def _fit_unsupervised(self, X: np.ndarray, hmm: ShipHMM) -> Any:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=4,
            covariance_type="full",
            n_iter=200,
        )
        model.fit(X)
        hmm.model = model
        hmm._transition_matrix = model.transmat_
        return model

    def _log_state_distribution(self, df: pd.DataFrame) -> None:
        counts = df["predicted_state"].value_counts()
        self.logger.info("HMM AIS state distribution:")
        for state, count in counts.items():
            pct = count / len(df) * 100
            self.logger.info("  %s: %d (%.1f%%)", state, count, pct)

    def _export_csv(self, df: pd.DataFrame) -> None:
        output = pd.DataFrame({"signaldate": df["signaldate"], "LAT": df["LAT"], "LON": df["LON"]})

        for state, col_name in STATE_COLUMNS.items():
            output[col_name] = (df["predicted_state"] == state).astype(int)

        output["AIS_SOG"] = df["sog_knots"]
        output["AIS_COG"] = df["cog_deg"]
        output["Confidence Rate"] = df["confidence"].round(1)

        if "destination_port" in df.columns:
            output["AIS_DESTINATION_PORT"] = df["destination_port"]

        path = self.reports_dir / "hmm_ais_output.csv"
        output.to_csv(path, index=False)
        self.logger.info("HMM AIS CSV exported to %s (%d rows)", path, len(output))

    def _compare_with_gps(self, ais_df: pd.DataFrame) -> dict:
        gps_reports = Path("reports")

        best_gps = None
        for exp_name in ["hmm_gps_v1", "rule_based_gps_v1"]:
            pred_path = gps_reports / exp_name / "test_predictions.csv"
            if pred_path.exists():
                best_gps = pred_path
                break
            full_path = gps_reports / exp_name / "output.csv"
            if full_path.exists():
                best_gps = full_path
                break

        if best_gps is None:
            self.logger.info("No GPS predictions found for comparison")
            return {}

        gps_df = pd.read_csv(best_gps)
        gps_df["signaldate"] = pd.to_datetime(gps_df["signaldate"])

        ais_start, ais_end = ais_df["signaldate"].min(), ais_df["signaldate"].max()
        gps_overlap = gps_df[
            (gps_df["signaldate"] >= ais_start) & (gps_df["signaldate"] <= ais_end)
        ]

        if len(gps_overlap) == 0:
            self.logger.info("No temporal overlap between AIS and GPS data")
            return {}

        gps_state_col = "predicted_state" if "predicted_state" in gps_overlap.columns else None
        if gps_state_col is None:
            for col in ["At Sea Voyage", "In Port", "At Sea Anchor", "At Sea Adrift GPS"]:
                if col not in gps_overlap.columns:
                    self.logger.info("GPS output missing expected columns")
                    return {}
            gps_overlap = gps_overlap.copy()
            gps_overlap["predicted_state"] = "anchor"
            gps_overlap.loc[gps_overlap["In Port"] == 1, "predicted_state"] = "port_stay"
            gps_overlap.loc[gps_overlap["At Sea Voyage"] == 1, "predicted_state"] = "voyage"
            gps_overlap.loc[gps_overlap["At Sea Anchor"] == 1, "predicted_state"] = "anchor"
            gps_overlap.loc[gps_overlap["At Sea Adrift GPS"] == 1, "predicted_state"] = "adrift"
            gps_state_col = "predicted_state"

        merged = pd.merge_asof(
            ais_df[["signaldate", "predicted_state", "sog_knots", "LAT", "LON"]].sort_values("signaldate"),
            gps_overlap[["signaldate", gps_state_col]].sort_values("signaldate"),
            on="signaldate",
            direction="nearest",
            tolerance=pd.Timedelta("5min"),
            suffixes=("_ais", "_gps"),
        )

        matched = merged.dropna(subset=[f"{gps_state_col}_gps"])
        if len(matched) == 0:
            self.logger.info("No matched points within 5min tolerance")
            return {}

        agreement = (matched["predicted_state_ais"] == matched[f"{gps_state_col}_gps"]).mean()
        self.logger.info(
            "GPS-AIS HMM comparison: %d matched points, %.1f%% agreement",
            len(matched), agreement * 100,
        )

        comparison_df = matched[["signaldate", "LAT", "LON", "sog_knots", "predicted_state_ais", f"{gps_state_col}_gps"]].copy()
        comparison_df.columns = ["signaldate", "LAT", "LON", "AIS_SOG", "ais_state", "gps_state"]
        comparison_df.to_csv(self.reports_dir / "gps_ais_hmm_comparison.csv", index=False)

        return {
            "matched_points": len(matched),
            "agreement_rate": round(agreement, 4),
        }
