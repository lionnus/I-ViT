#!/usr/bin/env python
"""
analyze_io_stats_simple.py  -  concise integer-range summary
===========================================================
Reads the CSV written by  save_io_stats_df(..., to_csv=True)  
containing integer extrema and shapes plus separate QuantMatMul and QuantLinear fields.
"""

from __future__ import annotations
import argparse, sys, ast
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _parse_tuple(col: pd.Series) -> pd.Series:
    """Safely turn a column that contains stringified tuples back into tuples."""
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def _shape_counts(series: pd.Series) -> str:
    """Return an indented, human-readable list of shape occurrences."""
    counts = series.value_counts().sort_index()
    lines = [f"    {shape}: {cnt}" for shape, cnt in counts.items()]
    return "\n".join(lines) if lines else "    - none"

# ---------------------------------------------------------------------------
#  Main logic
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, out: Path):
    out_lines: list[str] = []

    # Ensure shape columns are true tuples so they hash nicely
    for col in ("shape_in", "shape_out", "shape_A", "shape_B", "shape_weight"):
        if col in df.columns:
            df[col] = _parse_tuple(df[col])

    # ── Global extrema ────────────────────────────────────────────────────
    out_lines.append("GLOBAL INTEGER EXTREMA (all layer calls)")
    minima = df[["min_in_int", "min_out_int"]].min().to_dict()
    maxima = df[["max_in_int", "max_out_int"]].max().to_dict()
    out_lines.extend([
        f"  min_in_int : {minima['min_in_int']}",
        f"  max_in_int : {maxima['max_in_int']}",
        f"  min_out_int: {minima['min_out_int']}",
        f"  max_out_int: {maxima['max_out_int']}",
    ])
    out_lines.append("")

    # ── Shape occurrences (input tensors) ─────────────────────────────────
    out_lines.append("INPUT SHAPE OCCURRENCES (shape_in)")
    out_lines.append(_shape_counts(df["shape_in"]))
    out_lines.append("")

    # ── Per–quant-module stats ────────────────────────────────────────────
    # List every special module you want reported
    special_types = [
        "QuantAct",
        "QuantConv2d",
        "IntLayerNorm",
        "IntGELU",
        "IntSoftmax",
    ]
    for typ in special_types:
        sub = df[df["type"] == typ]
        if sub.empty:
            continue
        out_lines.append(f"{typ.upper()} (stats across all calls)")
        out_lines.append(f"  scale_in_min : {sub['scale_in'].min()}")
        out_lines.append(f"  scale_in_max : {sub['scale_in'].max()}")
        out_lines.append(f"  scale_out_min: {sub['scale_out'].min()}")
        out_lines.append(f"  scale_out_max: {sub['scale_out'].max()}")
        out_lines.append(f"  min_in_int : {sub['min_in_int'].min()}")
        out_lines.append(f"  max_in_int : {sub['max_in_int'].max()}")
        out_lines.append(f"  min_out_int: {sub['min_out_int'].min()}")
        out_lines.append(f"  max_out_int: {sub['max_out_int'].max()}")
        out_lines.append("  shapes_in :")
        out_lines.append(_shape_counts(sub["shape_in"]))
        out_lines.append("  shapes_out:")
        out_lines.append(_shape_counts(sub["shape_out"]))
        out_lines.append("")

    # ── QuantMatMul special case ─────────────────────────────────────────
    qmm = df[df["type"] == "QuantMatMul"]
    if not qmm.empty:
        out_lines.append("QUANTMATMUL INPUT A (stats across all calls)")
        out_lines.append(f"  min_A_int : {qmm['min_A_int'].min()}")
        out_lines.append(f"  max_A_int : {qmm['max_A_int'].max()}")
        out_lines.append("  shapes_A :")
        out_lines.append(_shape_counts(qmm["shape_A"]))
        out_lines.append("")
        out_lines.append("QUANTMATMUL INPUT B (stats across all calls)")
        out_lines.append(f"  min_B_int : {qmm['min_B_int'].min()}")
        out_lines.append(f"  max_B_int : {qmm['max_B_int'].max()}")
        out_lines.append("  shapes_B :")
        out_lines.append(_shape_counts(qmm["shape_B"]))
        out_lines.append("")
    else:
        out_lines.append("No QuantMatMul layers found - skipping A/B stats.\n")

    # ── QuantLinear weight stats (if present) ─────────────────────────────
    ql = df[df["type"] == "QuantLinear"]
    if not ql.empty and "min_weight_int" in df.columns:
        out_lines.append("QUANTLINEAR WEIGHT (stats across all calls)")
        out_lines.append(f"  min_weight_int : {ql['min_weight_int'].min()}")
        out_lines.append(f"  max_weight_int : {ql['max_weight_int'].max()}")
        if "shape_weight" in ql.columns:
            out_lines.append("  shapes_weight :")
            out_lines.append(_shape_counts(ql["shape_weight"]))
        out_lines.append("")

    # Write to disk & echo to console
    out.write_text("\n".join(out_lines))
    print("Summary written →", out.resolve())


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="CSV produced by save_io_stats_df(..., to_csv=True)")
    p.add_argument("--out", default=None,
                   help="Output text file (default: <csv_stem>_summary.txt)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERR: {csv_path} not found")

    df = pd.read_csv(csv_path)
    out_path = Path(args.out) if args.out else csv_path.with_suffix("").with_name(csv_path.stem + "_summary.txt")

    summarise(df, out_path)

if __name__ == "__main__":
    main()
