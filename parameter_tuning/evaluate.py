"""evaluate.py — ingest tracker run outputs and score against ground truth.

Usage:
    python parameter_tuning/evaluate.py

Edit RUN_MAP at the top to point at your latest run directories, then run.
Steps:
  1. Copy ordered_tracks.csv + detections_raw.csv from each run into data/
  2. Build MOT-format files (GT + tracker) for evaluation
  3. Run HOTA, CLEAR (MOTA/MOTP/IDSW), and Identity (IDF1) via trackers.eval
  4. Print a combined summary table
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from trackers.eval.evaluate import evaluate_mot_sequence

# ── Edit these to point at your latest run directories ──────────────────────
RUN_MAP = {
    "13d_002": Path(r"C:\Users\emmav\Downloads\superfly\outputs\run_112"),
    "31d_005": Path(r"C:\Users\emmav\Downloads\superfly\outputs\run_114_31DPE_n005"),
}
# ────────────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
OUT_ROOT     = ROOT / "results" / "mot_inputs"
TRACKER_NAME = "new"
SEQUENCES    = list(RUN_MAP.keys())


# ---------------------------------------------------------------------------
# Step 1 — ingest run outputs into data/
# ---------------------------------------------------------------------------

def _ingest(seq: str, run_dir: Path) -> None:
    for src_name, dst_name in [
        ("ordered_tracks.csv", f"tracks_baseline_{seq}.csv"),
        ("detections_raw.csv", f"detections_raw_{seq}.csv"),
    ]:
        src = run_dir / src_name
        dst = DATA_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(
                f"Missing {src}\n"
                f"(ordered_tracks.csv requires the vial assignment cell to have run)"
            )
        shutil.copy2(src, dst)
        print(f"  {src_name} -> data/{dst_name}")


# ---------------------------------------------------------------------------
# Step 2 — build MOT files
# ---------------------------------------------------------------------------

def _mot_lines(df: pd.DataFrame, *, c7: float) -> list[str]:
    w = df["x2"] - df["x1"]
    h = df["y2"] - df["y1"]
    if (w <= 0).any() or (h <= 0).any():
        bad = df[(w <= 0) | (h <= 0)]
        raise ValueError(f"non-positive bbox dims in {len(bad)} rows:\n{bad.head()}")
    out = pd.DataFrame({
        "frame": df["frame"].astype(int) + 1,   # 1-indexed for MOT
        "id":    df["id"].astype(int),
        "bbl":   df["x1"].astype(float),
        "bbt":   df["y1"].astype(float),
        "bbw":   w.astype(float),
        "bbh":   h.astype(float),
        "c7":    c7,
        "c8":    1,
        "c9":    1.0,
    })
    return [
        f"{r.frame},{r.id},{r.bbl:.3f},{r.bbt:.3f},{r.bbw:.3f},{r.bbh:.3f},"
        f"{r.c7},{r.c8},{r.c9}"
        for r in out.itertuples(index=False)
    ]


def _build_gt(seq: str) -> list[str]:
    df = pd.read_csv(DATA_DIR / f"ground_truth_{seq}.csv")
    df = df.rename(columns={"ID": "id"})
    return _mot_lines(df[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1)


def _build_tracker(seq: str) -> list[str]:
    tracks = pd.read_csv(DATA_DIR / f"tracks_baseline_{seq}.csv")
    dets   = pd.read_csv(DATA_DIR / f"detections_raw_{seq}.csv")

    tracks = tracks.dropna(subset=["ordered_id"]).copy()
    tracks["ordered_id"] = tracks["ordered_id"].astype(int)

    dets = dets.copy()
    dets["x"] = (dets["x1"] + dets["x2"]) / 2.0
    dets["y"] = (dets["y1"] + dets["y2"]) / 2.0
    n_before = len(dets)
    dets = dets.sort_values("conf", ascending=False).drop_duplicates(
        subset=["frame", "x", "y"], keep="first"
    )
    if len(dets) < n_before:
        print(f"  [{seq}] dedup: dropped {n_before - len(dets)} duplicate detections")

    merged = tracks.merge(
        dets[["frame", "x", "y", "x1", "y1", "x2", "y2"]],
        on=["frame", "x", "y"], how="left", validate="many_to_one",
    )
    unmatched = merged[merged["x1"].isna()]
    if len(unmatched):
        raise RuntimeError(
            f"{len(unmatched)} tracker rows in {seq} failed to join to a detection.\n"
            f"First few:\n{unmatched[['frame', 'ordered_id', 'x', 'y']].head()}"
        )
    merged = merged.rename(columns={"ordered_id": "id"})
    return _mot_lines(merged[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1.0)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_mot_files() -> None:
    for seq in SEQUENCES:
        gt_lines = _build_gt(seq)
        tr_lines = _build_tracker(seq)
        gt_path  = OUT_ROOT / "gt" / seq / "gt" / "gt.txt"
        tr_path  = OUT_ROOT / "trackers" / TRACKER_NAME / "data" / f"{seq}.txt"
        _write(gt_path, gt_lines)
        _write(tr_path, tr_lines)
        print(f"  {seq}: {len(gt_lines)} GT rows, {len(tr_lines)} tracker rows")


# ---------------------------------------------------------------------------
# Step 3 — evaluate with trackers.eval (HOTA + CLEAR + Identity)
# ---------------------------------------------------------------------------

def _run_eval() -> None:
    col_w = max(len(s) for s in SEQUENCES + ["COMBINED"])
    header = (
        f"{'video':<{col_w}}  "
        f"{'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  "
        f"{'MOTA':>6}  {'MOTP':>6}  {'IDSW':>5}  "
        f"{'IDF1':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    seq_results = {}
    for seq in SEQUENCES:
        gt_path  = OUT_ROOT / "gt" / seq / "gt" / "gt.txt"
        tr_path  = OUT_ROOT / "trackers" / TRACKER_NAME / "data" / f"{seq}.txt"
        result   = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tr_path,
            metrics=["HOTA", "CLEAR", "Identity"],
        )
        seq_results[seq] = result

        h = result.HOTA
        c = result.CLEAR
        i = result.Identity
        print(
            f"{seq:<{col_w}}  "
            f"{h.HOTA:>6.3f}  {h.DetA:>6.3f}  {h.AssA:>6.3f}  "
            f"{c.MOTA:>6.3f}  {c.MOTP:>6.3f}  {int(c.IDSW):>5d}  "
            f"{i.IDF1:>6.3f}"
        )

    # Combined (simple mean across sequences for float metrics, sum for IDSW)
    if len(SEQUENCES) > 1:
        hota_avg = sum(seq_results[s].HOTA.HOTA for s in SEQUENCES) / len(SEQUENCES)
        deta_avg = sum(seq_results[s].HOTA.DetA for s in SEQUENCES) / len(SEQUENCES)
        assa_avg = sum(seq_results[s].HOTA.AssA for s in SEQUENCES) / len(SEQUENCES)
        mota_avg = sum(seq_results[s].CLEAR.MOTA for s in SEQUENCES) / len(SEQUENCES)
        motp_avg = sum(seq_results[s].CLEAR.MOTP for s in SEQUENCES) / len(SEQUENCES)
        idsw_sum = sum(int(seq_results[s].CLEAR.IDSW) for s in SEQUENCES)
        idf1_avg = sum(seq_results[s].Identity.IDF1 for s in SEQUENCES) / len(SEQUENCES)
        print("-" * len(header))
        print(
            f"{'COMBINED':<{col_w}}  "
            f"{hota_avg:>6.3f}  {deta_avg:>6.3f}  {assa_avg:>6.3f}  "
            f"{mota_avg:>6.3f}  {motp_avg:>6.3f}  {idsw_sum:>5d}  "
            f"{idf1_avg:>6.3f}"
        )


# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Step 1: ingesting run outputs ===")
    for seq, run_dir in RUN_MAP.items():
        print(f"  {seq}  <-  {run_dir.name}")
        _ingest(seq, run_dir)

    print("\n=== Step 2: building MOT files ===")
    _build_mot_files()

    print("\n=== Step 3: evaluating (HOTA + CLEAR + Identity) ===")
    _run_eval()


if __name__ == "__main__":
    main()
