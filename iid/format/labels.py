from pathlib import Path

import polars as pl

LABELS = {
    (_ + 1): label
    for _, label in enumerate(
        pl.read_csv(str(Path(__file__).parent / "labels.csv"), has_header=False)[
            "column_1"
        ]
    )
}
