from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ais_preprocessing import AISPreprocessor
from src.data.dataset import load_ais_csv, load_port_database
from src.data.ports import PortMatcher
from src.experiments.registry import register_experiment
from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)

EXTENDED_STATES = [
    "port_stay",
    "port_maneuver",
    "voyage",
    "anchor",
    "adrift",
    "turn",
    "shifting",
]

EXTENDED_STATE_COLUMNS = {
    "port_stay": "In Port",
    "port_maneuver": "Port Maneuver",
    "voyage": "At Sea Voyage",
    "anchor": "At Sea Anchor",
    "adrift": "At Sea Adrift",
    "turn": "Turn",
    "shifting": "Shifting",
}


class AISRuleClassifier:
    """Rule-based classifier for AIS data with extended states."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.voyage_speed_kn: float = p.get("voyage_speed_kn", 3.0)
        self.maneuver_speed_kn: float = p.get("maneuver_speed_kn", 2.0)
        self.stationary_speed_kn: float = p.get("stationary_speed_kn", 0.5)
        self.turn_dcog_threshold: float = p.get("turn_dcog_threshold", 30.0)
        self.drift_angle_threshold: float = p.get("drift_angle_threshold", 15.0)
        self.shifting_speed_min_kn: float = p.get("shifting_speed_min_kn", 1.0)
        self.shifting_speed_max_kn: float = p.get("shifting_speed_max_kn", 5.0)
        self.anchor_spread_max_m: float = p.get("anchor_spread_max_m", 200.0)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        states = np.empty(n, dtype=object)
        confidences = np.zeros(n)

        sog = df["sog_knots"].values
        dcog = np.abs(df["dcog_deg"].fillna(0).values)
        in_port = df["in_port"].values if "in_port" in df.columns else np.zeros(n, dtype=bool)
        drift_angle = np.abs(df["drift_angle_deg"].fillna(0).values) if "drift_angle_deg" in df.columns else np.zeros(n)

        for i in range(n):
            s = sog[i]
            dc = dcog[i]
            da = drift_angle[i]
            ip = in_port[i]

            if ip and s < self.stationary_speed_kn:
                states[i] = "port_stay"
                confidences[i] = min(95.0, 50 + (self.stationary_speed_kn - s) * 80)
            elif ip and self.stationary_speed_kn <= s <= self.maneuver_speed_kn:
                states[i] = "port_maneuver"
                confidences[i] = 60 + min(30, s * 10)
            elif s >= self.voyage_speed_kn:
                if dc > self.turn_dcog_threshold:
                    states[i] = "turn"
                    confidences[i] = min(95.0, 50 + dc)
                else:
                    states[i] = "voyage"
                    confidences[i] = min(95.0, 50 + s * 5)
            elif self.shifting_speed_min_kn <= s <= self.shifting_speed_max_kn and ip:
                states[i] = "shifting"
                confidences[i] = 55 + min(35, s * 10)
            elif s < self.stationary_speed_kn:
                states[i] = "anchor"
                confidences[i] = min(90.0, 50 + (self.stationary_speed_kn - s) * 80)
            elif da > self.drift_angle_threshold:
                states[i] = "adrift"
                confidences[i] = min(90.0, 50 + da)
            else:
                states[i] = "adrift"
                confidences[i] = 40.0

        df["predicted_state"] = states
        df["confidence"] = confidences
        return df


@register_experiment("extended_ais")
class ExtendedAISExperiment:
    """Variant B: GPS+AIS with extended states and comparative analysis."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = create_logger(config.experiment_name)
        _, _, self.reports_dir = config.experiment_paths()

    @classmethod
    def name(cls) -> str:
        return "extended_ais"

    def run(self) -> dict:
        ais_file = self.config.model.params.get("ais_file", "Ship_Operation_example_dateset_AIS_based_data.csv")
        ais_path = Path(self.config.paths.raw_dir).parent / "unclassified" / ais_file

        ais_df = load_ais_csv(ais_path)

        preprocessor = AISPreprocessor()
        ais_df = preprocessor.process(ais_df)

        ports_df = load_port_database(self.config)
        matcher = PortMatcher(ports_df, self.config.port)
        port_info = matcher.query_batch(ais_df["LAT"].values, ais_df["LON"].values)
        ais_df = ais_df.join(port_info)

        classifier = AISRuleClassifier(self.config.model.params)
        ais_classified = classifier.classify(ais_df)

        self._log_state_distribution(ais_classified)

        ais_classified.to_csv(self.reports_dir / "ais_classified.csv", index=False)
        self._export_extended_csv(ais_classified)

        comparison = self._compare_with_gps(ais_classified)

        results = {
            "ais_rows": len(ais_classified),
            "state_distribution": ais_classified["predicted_state"].value_counts().to_dict(),
            "comparison": comparison,
        }

        self.logger.info("Extended AIS experiment complete")
        return results

    def _log_state_distribution(self, df: pd.DataFrame) -> None:
        counts = df["predicted_state"].value_counts()
        self.logger.info("AIS state distribution:")
        for state, count in counts.items():
            pct = count / len(df) * 100
            self.logger.info("  %s: %d (%.1f%%)", state, count, pct)

    def _export_extended_csv(self, df: pd.DataFrame) -> None:
        output = pd.DataFrame({"signaldate": df["signaldate"], "LAT": df["LAT"], "LON": df["LON"]})

        for state, col_name in EXTENDED_STATE_COLUMNS.items():
            output[col_name] = (df["predicted_state"] == state).astype(int)

        output["AIS_SOG"] = df["sog_knots"]
        output["AIS_COG"] = df["cog_deg"]
        output["AIS_HDG"] = df.get("hdg_deg", df["cog_deg"])
        output["Confidence Rate"] = df["confidence"].round(1)

        if "destination_port" in df.columns:
            output["AIS_DESTINATION_PORT"] = df["destination_port"]

        path = self.reports_dir / "extended_output.csv"
        output.to_csv(path, index=False)
        self.logger.info("Extended CSV exported to %s (%d rows)", path, len(output))

    def _compare_with_gps(self, ais_df: pd.DataFrame) -> dict:
        """Compare AIS classification with GPS ground truth on overlapping time range."""
        gps_reports = Path("reports")

        best_gps = None
        for exp_name in ["rule_based_gps_v1"]:
            pred_path = gps_reports / exp_name / "test_predictions.csv"
            if pred_path.exists():
                best_gps = pred_path
                break

        if best_gps is None:
            full_path = gps_reports / "rule_based_gps_v1" / "output.csv"
            if full_path.exists():
                best_gps = full_path

        if best_gps is None:
            self.logger.info("No GPS predictions found for comparison")
            return {}

        gps_df = pd.read_csv(best_gps)
        gps_df["signaldate"] = pd.to_datetime(gps_df["signaldate"])

        ais_start, ais_end = ais_df["signaldate"].min(), ais_df["signaldate"].max()
        gps_overlap = gps_df[(gps_df["signaldate"] >= ais_start) & (gps_df["signaldate"] <= ais_end)]

        if len(gps_overlap) == 0:
            self.logger.info("No temporal overlap between AIS and GPS data")
            return {}

        state_map_ais_to_basic = {
            "port_stay": "port_stay",
            "port_maneuver": "port_stay",
            "voyage": "voyage",
            "anchor": "anchor",
            "adrift": "adrift",
            "turn": "voyage",
            "shifting": "port_stay",
        }

        ais_basic = ais_df.copy()
        ais_basic["basic_state"] = ais_basic["predicted_state"].map(state_map_ais_to_basic)

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
            ais_basic.sort_values("signaldate"),
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

        agreement = (matched["basic_state"] == matched[f"{gps_state_col}_gps"]).mean()

        self.logger.info("GPS-AIS comparison: %d matched points, %.1f%% agreement", len(matched), agreement * 100)

        comparison_df = matched[["signaldate", "LAT", "LON", "sog_knots", "predicted_state_ais", "basic_state", f"{gps_state_col}_gps"]].copy()
        comparison_df.columns = ["signaldate", "LAT", "LON", "AIS_SOG", "ais_extended_state", "ais_basic_state", "gps_state"]
        comparison_df.to_csv(self.reports_dir / "gps_ais_comparison.csv", index=False)

        return {
            "matched_points": len(matched),
            "agreement_rate": round(agreement, 4),
            "ais_states": matched["predicted_state_ais"].value_counts().to_dict(),
            "gps_states": matched[f"{gps_state_col}_gps"].value_counts().to_dict(),
        }
