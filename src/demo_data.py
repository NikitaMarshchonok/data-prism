"""Deterministic, privacy-safe demonstration data for Data Prism."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEMO_DATASET_ID = "saas_growth"
DEMO_FILENAME = "saas_growth_demo.csv"
DEMO_PROMPT = (
    "Create an evidence-first SaaS growth dashboard with revenue and customer KPIs, "
    "regional and plan comparisons, trends, anomalies, statistical validation, and "
    "clear next-step recommendations."
)


def create_saas_growth_demo(row_count: int = 360, *, seed: int = 42) -> pd.DataFrame:
    """Create a repeatable SaaS operating dataset without personal information."""
    if not 90 <= row_count <= 5000:
        raise ValueError("row_count must be between 90 and 5000.")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=row_count, freq="D")
    day = np.arange(row_count)

    regions = rng.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America"],
        size=row_count,
        p=[0.36, 0.29, 0.23, 0.12],
    )
    plans = rng.choice(
        ["Starter", "Growth", "Enterprise"],
        size=row_count,
        p=[0.43, 0.39, 0.18],
    )
    channels = rng.choice(
        ["Organic", "Partner", "Paid Search", "Events"],
        size=row_count,
        p=[0.38, 0.27, 0.25, 0.10],
    )

    plan_scale = pd.Series(plans).map(
        {"Starter": 0.72, "Growth": 1.08, "Enterprise": 1.72}
    ).to_numpy()
    region_scale = pd.Series(regions).map(
        {
            "North America": 1.18,
            "Europe": 1.04,
            "Asia Pacific": 0.94,
            "Latin America": 0.78,
        }
    ).to_numpy()
    channel_scale = pd.Series(channels).map(
        {"Organic": 1.08, "Partner": 1.13, "Paid Search": 0.94, "Events": 1.02}
    ).to_numpy()

    weekly_cycle = 1 + 0.07 * np.sin(2 * np.pi * day / 7)
    growth = 1 + day * 0.0015
    customer_count = np.maximum(
        25,
        np.rint(
            92 * plan_scale * region_scale * growth * weekly_cycle
            + rng.normal(0, 8, row_count)
        ),
    ).astype(int)

    conversion_rate = np.clip(
        0.035 * plan_scale * channel_scale
        + 0.004 * np.sin(2 * np.pi * day / 30)
        + rng.normal(0, 0.004, row_count),
        0.008,
        0.14,
    )
    churn_rate = np.clip(
        0.072
        - 0.019 * (plans == "Enterprise")
        - 0.011 * (plans == "Growth")
        + 0.012 * (channels == "Paid Search")
        + rng.normal(0, 0.007, row_count),
        0.012,
        0.14,
    )
    average_revenue = pd.Series(plans).map(
        {"Starter": 39.0, "Growth": 92.0, "Enterprise": 248.0}
    ).to_numpy()
    monthly_revenue = (
        customer_count
        * average_revenue
        * region_scale
        * (1 + rng.normal(0, 0.035, row_count))
    )
    ticket_count = rng.poisson(
        np.maximum(1.0, customer_count * (0.055 + churn_rate * 0.7))
    ).astype(float)
    nps_score = np.clip(
        67
        + 90 * (conversion_rate - conversion_rate.mean())
        - 150 * (churn_rate - churn_rate.mean())
        - 0.28 * (ticket_count - ticket_count.mean())
        + rng.normal(0, 5, row_count),
        0,
        100,
    )

    # Deterministic incidents make the anomaly and evidence panels meaningful.
    incident_rows = np.array([57, 181, 294])
    incident_rows = incident_rows[incident_rows < row_count]
    monthly_revenue[incident_rows] *= np.array([0.48, 1.85, 0.55])[: len(incident_rows)]
    ticket_count[incident_rows] *= np.array([4.2, 3.4, 4.8])[: len(incident_rows)]
    churn_rate[incident_rows] = np.clip(churn_rate[incident_rows] * 1.9, 0, 0.2)

    missing_rows = np.arange(23, row_count, 47)
    nps_score[missing_rows] = np.nan

    renewed = np.where(
        rng.random(row_count) > np.clip(churn_rate * 3.2, 0.05, 0.55),
        "Yes",
        "No",
    )

    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Region": regions,
            "Plan": plans,
            "AcquisitionChannel": channels,
            "CustomerCount": customer_count,
            "MonthlyRevenue": monthly_revenue.round(2),
            "ConversionRate": conversion_rate.round(4),
            "ChurnRate": churn_rate.round(4),
            "TicketCount": ticket_count,
            "NPSScore": nps_score.round(1),
            "Renewed": renewed,
        }
    )


def save_demo_dataset(output_path: str | Path, *, row_count: int = 360) -> Path:
    """Generate and persist the demonstration dataset."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    create_saas_growth_demo(row_count=row_count).to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Data Prism SaaS demo dataset."
    )
    parser.add_argument(
        "--output",
        default=f"data/demo/{DEMO_FILENAME}",
        help="CSV output path (default: %(default)s).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=360,
        help="Number of rows between 90 and 5000 (default: %(default)s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = save_demo_dataset(args.output, row_count=args.rows)
    except (OSError, ValueError) as error:
        print(f"Could not generate demo dataset: {error}")
        return 1
    print(f"Generated {args.rows:,} synthetic rows at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
