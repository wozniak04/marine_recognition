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
from src.models.ais_classifier import AIS_TO_BASIC, AISRuleClassifier, EXTENDED_STATES
from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)

EXTENDED_STATE_COLUMNS = {
    "port_stay": "In Port",
    "port_maneuver": "Port Maneuver",
    "voyage": "At Sea Voyage",
    "anchor": "At Sea Anchor",
    "adrift": "At Sea Adrift",
    "turn": "Turn",
    "shifting": "Shifting",
}


@register_experiment("extended_ais")
class ExtendedAISExperiment:

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

        teleport_mask = detect_teleports(ais_df, max_speed_knots=self.config.outlier.max_speed_knots)
        if teleport_mask.any():
            ais_df = fix_teleports(ais_df, teleport_mask)

        ports_df = load_port_database(self.config)
        discovered = discover_ports(ais_df)
        matcher = build_port_matcher(ports_df, discovered, self.config.port)
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

        ais_basic = ais_df.copy()
        ais_basic["basic_state"] = ais_basic["predicted_state"].map(AIS_TO_BASIC)

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
