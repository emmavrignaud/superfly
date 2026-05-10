"""Merge two labeler annotation sessions for the 31d_005 video.

User annotated vials 1 & 3, teammate annotated vials 2, 4, 5, 6 — independently,
on top of two different OCSort runs (so their track IDs are NOT comparable).
Both sessions share byte-identical `detections_raw.csv`, so the
`(frame, det_idx)` keys are interchangeable for *real* detections.

Spatial owner-wins rule: each detection is assigned to one vial via its (x, y);
only the vial owner's annotation survives. Both `human` and `ocsort`-sourced
annotations are kept (the labeler treats ocsort-sourced as accepted ground
truth unless cleared).

Synthetic detections (detector misses re-added by hand, negative det_idx) are
preserved. The user's synthetic det_idx range is shifted to avoid collision
with the teammate's, and the user's annotation keys referencing them are
re-keyed accordingly.

Track IDs from the user's session are shifted by `max(teammate_id) + 1` so the
two sessions' independent OCSort runs can coexist without renumbering chaos.

`confirmed_tracks` from both sessions are merged (user IDs remapped) and
trimmed to track IDs that actually survive the vial filter.

Output: a fresh per-video labeling folder under
`data/manual_labelling/combined_annotations/<videostem>/` containing the
merged `.labeler.json`, `gt.csv`, `gt_summary.txt`, video copy,
`detections_raw.csv` copy, and `tracks_long.csv`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `labeler` importable when running this script directly.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from labeler.assets import (  # noqa: E402
    init_metadata,
    populate_folder,
    write_export_summary,
)
from labeler.data_model import (  # noqa: E402
    Annotation,
    AnnotationStore,
    Detection,
    SOURCE_HUMAN,
    SOURCE_OCSORT,
    load_raw_detections,
)
from labeler.session import (  # noqa: E402
    SESSION_VERSION,
    _key_to_str,
    _str_to_key,
)


# ----------------------------- CONFIG --------------------------------------

VIDEO_STEM = (
    "2024-03-01_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m"
    "_31d_005-converted_raw_cropped"
)
ROI_LIBRARY_KEY = (
    "2024-03-01_NEG-008_hTDP43_WT-A90V-G287S-G294A-A315T-M337V_m_31d_005-converted"
)

# Each source session's per-video folder.
TEAMMATE_DIR = REPO / "data" / "manual_labelling" / VIDEO_STEM
USER_DIR = REPO / "Ahmed Annotations 31d_005" / VIDEO_STEM

# Vial ownership (1-indexed, matching roi_library.json keys).
USER_VIALS = {1, 3}
TEAMMATE_VIALS = {2, 4, 5, 6}

# Output folder.
OUT_DIR = REPO / "data" / "manual_labelling" / "combined_annotations" / VIDEO_STEM


# ----------------------------- helpers -------------------------------------

def _load_session(session_path: Path) -> dict:
    return json.loads(session_path.read_text(encoding="utf-8"))


def _load_vial_boxes(roi_library_path: Path) -> dict[int, tuple[int, int, int, int]]:
    """Return {vial_number: (x0, y0, x1, y1)} for the configured video."""
    lib = json.loads(roi_library_path.read_text(encoding="utf-8"))
    entry = lib[ROI_LIBRARY_KEY]
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for name, box in entry["vial_rois"].items():
        n = int(name.removeprefix("vial"))
        x0, y0, x1, y1 = box
        boxes[n] = (int(x0), int(y0), int(x1), int(y1))
    return boxes


def _vial_of(x: float, y: float, boxes: dict[int, tuple[int, int, int, int]]) -> int:
    for n, (x0, y0, x1, y1) in boxes.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return n
    raise AssertionError(f"detection at ({x:.2f}, {y:.2f}) is in no vial")


def _parse_synthetics(payload: dict) -> list[Detection]:
    """Read synthetic_detections array straight from a session payload as
    Detection objects (negative det_idx)."""
    out: list[Detection] = []
    for r in payload.get("synthetic_detections", []):
        out.append(Detection(
            frame=int(r["frame"]), det_idx=int(r["det_idx"]),
            x=float(r["x"]), y=float(r["y"]),
            x1=float(r["x1"]), y1=float(r["y1"]),
            x2=float(r["x2"]), y2=float(r["y2"]),
            conf=float("nan"), is_synthetic=True,
        ))
    return out


def _filter_annotations(
    annotations: dict,                     # raw JSON dict {key_str: {track_id, source}}
    det_index: dict[tuple[int, int], Detection],
    boxes: dict[int, tuple[int, int, int, int]],
    keep_vials: set[int],
    *,
    label: str,
    key_remap: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> dict[tuple[int, int], Annotation]:
    """Filter annotations to those whose detection (real or synthetic) lies in
    `keep_vials`. `key_remap` re-keys (frame, old_det_idx) → (frame, new_det_idx)
    for synthetic detections that have been renumbered."""
    out: dict[tuple[int, int], Annotation] = {}
    by_source = {SOURCE_HUMAN: 0, SOURCE_OCSORT: 0}
    n_dropped_not_my_vial = 0
    n_missing = 0
    n_synth_kept = 0

    for k, v in annotations.items():
        frame, det_idx = _str_to_key(k)
        # Apply re-keying for synthetics (negative det_idx).
        if key_remap and (frame, det_idx) in key_remap:
            frame_new, det_idx_new = key_remap[(frame, det_idx)]
        else:
            frame_new, det_idx_new = frame, det_idx
        det = det_index.get((frame_new, det_idx_new))
        if det is None:
            n_missing += 1
            continue
        vial = _vial_of(det.x, det.y, boxes)
        if vial not in keep_vials:
            n_dropped_not_my_vial += 1
            continue
        src = v.get("source", SOURCE_HUMAN)
        by_source[src] = by_source.get(src, 0) + 1
        if det.is_synthetic:
            n_synth_kept += 1
        out[(frame_new, det_idx_new)] = Annotation(
            track_id=int(v["track_id"]), source=src,
        )

    print(
        f"  [{label}] kept {len(out):>5}  "
        f"(human={by_source.get(SOURCE_HUMAN, 0)}, "
        f"ocsort={by_source.get(SOURCE_OCSORT, 0)}, "
        f"synthetic={n_synth_kept})  "
        f"dropped: in-someone-elses-vial={n_dropped_not_my_vial}, "
        f"unresolved-key={n_missing}"
    )
    return out


# ----------------------------- main ----------------------------------------

def main() -> int:
    print(f"repo root            : {REPO}")
    print(f"teammate dir         : {TEAMMATE_DIR}")
    print(f"user dir             : {USER_DIR}")
    print(f"out dir              : {OUT_DIR}")
    print(f"vial split           : user={sorted(USER_VIALS)}  teammate={sorted(TEAMMATE_VIALS)}")
    print()

    # 1. Load shared real detections (both source dirs are byte-identical).
    raw_csv = TEAMMATE_DIR / "detections_raw.csv"
    print(f"loading detections   : {raw_csv}")
    raw_by_frame = load_raw_detections(str(raw_csv))
    n_dets = sum(len(v) for v in raw_by_frame.values())
    print(f"  {n_dets} real detections across {len(raw_by_frame)} frames\n")

    # 2. Load ROI boxes for this video.
    boxes = _load_vial_boxes(REPO / "roi_library.json")
    print("vial boxes (x0,y0,x1,y1):")
    for n in sorted(boxes):
        owner = "user" if n in USER_VIALS else "teammate"
        print(f"  vial{n} ({owner:>8}): {boxes[n]}")
    print()

    # 3. Load both session payloads.
    teammate_session = _load_session(TEAMMATE_DIR / f"{VIDEO_STEM}.labeler.json")
    user_session = _load_session(USER_DIR / f"{VIDEO_STEM}.labeler.json")

    # 4. Read synthetic detections from each side.
    teammate_synth = _parse_synthetics(teammate_session)
    user_synth_orig = _parse_synthetics(user_session)
    print(f"synthetics in teammate session : {len(teammate_synth)}")
    print(f"synthetics in user session     : {len(user_synth_orig)}")

    # 5. Renumber user's synthetic det_idx to a non-overlapping negative range.
    teammate_synth_idxs = {d.det_idx for d in teammate_synth}
    user_synth_idxs = {d.det_idx for d in user_synth_orig}
    shift = 0
    if teammate_synth and user_synth_orig:
        # Shift so that user_max + shift < teammate_min  → all user idxs more
        # negative than every teammate idx.
        teammate_min = min(teammate_synth_idxs)   # most negative teammate idx
        user_max = max(user_synth_idxs)           # least-negative (largest) user idx
        # We want user_max + shift = teammate_min - 1
        shift = (teammate_min - 1) - user_max
    user_synth = [
        Detection(
            frame=d.frame, det_idx=d.det_idx + shift,
            x=d.x, y=d.y, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
            conf=d.conf, is_synthetic=True,
        )
        for d in user_synth_orig
    ]
    user_synth_key_remap: dict[tuple[int, int], tuple[int, int]] = {
        (d_orig.frame, d_orig.det_idx): (d_new.frame, d_new.det_idx)
        for d_orig, d_new in zip(user_synth_orig, user_synth)
        if d_orig.det_idx != d_new.det_idx
    }
    print(f"user synthetic det_idx shift   : {shift}  "
          f"(remapped {len(user_synth_key_remap)} keys)\n")

    # 6. Build a unified detection index covering real dets and BOTH sides'
    #    synthetics. (Synthetic det_idx is unique per side after renumbering.)
    det_index: dict[tuple[int, int], Detection] = {
        (f, d.det_idx): d for f, dets in raw_by_frame.items() for d in dets
    }
    for d in teammate_synth:
        det_index[(d.frame, d.det_idx)] = d
    for d in user_synth:
        det_index[(d.frame, d.det_idx)] = d

    # 7. Filter each session's annotations by vial ownership.
    print("filtering teammate session:")
    teammate_anns = _filter_annotations(
        teammate_session.get("annotations", {}),
        det_index, boxes,
        keep_vials=TEAMMATE_VIALS, label="teammate",
    )
    print("filtering user session:")
    user_anns = _filter_annotations(
        user_session.get("annotations", {}),
        det_index, boxes,
        keep_vials=USER_VIALS, label="user",
        key_remap=user_synth_key_remap,
    )
    print()

    # 8. Remap user track IDs to a non-colliding range.
    teammate_ids = sorted({a.track_id for a in teammate_anns.values()})
    user_ids = sorted({a.track_id for a in user_anns.values()})
    if not teammate_ids:
        raise RuntimeError("teammate produced zero kept annotations — check vials")
    track_offset = max(teammate_ids) + 1
    track_remap = {old: old + track_offset for old in user_ids}
    user_anns_remapped = {
        k: Annotation(track_id=track_remap[a.track_id], source=a.source)
        for k, a in user_anns.items()
    }

    print(f"teammate distinct IDs: {len(teammate_ids)}  range=[{teammate_ids[0]}, {teammate_ids[-1]}]")
    print(f"user distinct IDs    : {len(user_ids)}  range=[{user_ids[0]}, {user_ids[-1]}]")
    print(f"user IDs shifted by +{track_offset}\n")

    # 9. Combine. With disjoint vials and renumbered synthetic keys, the two
    #    annotation maps must have disjoint keys.
    overlap = set(teammate_anns) & set(user_anns_remapped)
    if overlap:
        raise AssertionError(
            f"{len(overlap)} (frame, det_idx) keys overlap after vial split — "
            f"someone annotated outside their lane. Examples: {sorted(overlap)[:5]}"
        )
    combined: dict[tuple[int, int], Annotation] = {**teammate_anns, **user_anns_remapped}

    new_user_ids = {a.track_id for a in user_anns_remapped.values()}
    if set(teammate_ids) & new_user_ids:
        raise AssertionError("track-ID collision after remapping — bug in offset logic")

    # 10. Trim synthetics to the ones whose annotation actually survived.
    used_synth_keys = {(f, d) for (f, d) in combined if d < 0}
    surviving_synth = [
        d for d in (teammate_synth + user_synth)
        if (d.frame, d.det_idx) in used_synth_keys
    ]
    n_synth_dropped = (len(teammate_synth) + len(user_synth)) - len(surviving_synth)
    print(f"synthetics surviving merge     : {len(surviving_synth)}  "
          f"(dropped {n_synth_dropped} whose annotation didn't make the cut)")

    # 11. Merge confirmed_tracks (apply track remap to user side; trim to IDs
    #     that survive in the combined annotations).
    teammate_conf = [int(t) for t in teammate_session.get("confirmed_tracks", [])]
    user_conf = [int(t) for t in user_session.get("confirmed_tracks", [])]
    user_conf_remapped = [track_remap.get(t, t + track_offset) for t in user_conf]
    final_track_ids = {a.track_id for a in combined.values()}
    confirmed = sorted({
        t for t in (teammate_conf + user_conf_remapped) if t in final_track_ids
    })
    print(f"confirmed_tracks (teammate)    : {len(teammate_conf)}")
    print(f"confirmed_tracks (user)        : {len(user_conf)}")
    print(f"confirmed_tracks merged & kept : {len(confirmed)}\n")

    print(f"combined annotations : {len(combined)}")
    print(f"combined distinct IDs: {len(final_track_ids)}\n")

    # 12. Per-vial breakdown (sanity).
    by_vial: dict[int, int] = {n: 0 for n in range(1, 7)}
    for (f, di), _ann in combined.items():
        det = det_index[(f, di)]
        by_vial[_vial_of(det.x, det.y, boxes)] += 1
    print("per-vial annotation counts in combined output:")
    for n in sorted(by_vial):
        owner = "user" if n in USER_VIALS else "teammate"
        print(f"  vial{n} ({owner:>8}): {by_vial[n]}")
    print()

    # 13. Materialise output folder via the labeler's own helpers.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_video = TEAMMATE_DIR / f"{VIDEO_STEM}.mp4"
    src_tracks_long = TEAMMATE_DIR / "tracks_long.csv"
    if not src_video.exists():
        raise FileNotFoundError(f"video not found: {src_video}")
    if not src_tracks_long.exists():
        raise FileNotFoundError(f"tracks_long.csv not found: {src_tracks_long}")

    copy_actions = populate_folder(
        OUT_DIR,
        video_path=src_video,
        raw_csv=raw_csv,
        tracks_csv=src_tracks_long,
    )
    print(f"populate_folder      : {copy_actions}\n")

    teammate_md = json.loads((TEAMMATE_DIR / "metadata.json").read_text(encoding="utf-8"))
    video_props = teammate_md.get("video", {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0})

    init_metadata(
        OUT_DIR,
        repo_root=REPO,
        video_path=src_video,
        raw_csv=raw_csv,
        tracks_csv=src_tracks_long,
        video_props=video_props,
        copy_actions=copy_actions,
    )

    # 14. Write the merged .labeler.json with surviving synthetics + confirmed.
    session_path = OUT_DIR / f"{VIDEO_STEM}.labeler.json"
    payload = {
        "version": SESSION_VERSION,
        "video_path": (OUT_DIR / src_video.name).as_posix(),
        "raw_csv": (OUT_DIR / "detections_raw.csv").as_posix(),
        "ocsort_csv": (OUT_DIR / "tracks_long.csv").as_posix(),
        "current_frame": 0,
        "current_mode": "track",
        "annotations": {
            _key_to_str(f, d): {"track_id": int(a.track_id), "source": a.source}
            for (f, d), a in combined.items()
        },
        "synthetic_detections": [
            {
                "frame": d.frame, "det_idx": d.det_idx,
                "x": d.x, "y": d.y,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            }
            for d in surviving_synth
        ],
        "confirmed_tracks": confirmed,
    }
    session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote session        : {session_path}")

    # 15. Export gt.csv and summary using the same code paths the GUI uses.
    #     Build a raw_by_frame that includes the surviving synthetics so the
    #     AnnotationStore lookup (which searches by det_idx field) finds them.
    raw_by_frame_with_synth = {f: list(dets) for f, dets in raw_by_frame.items()}
    for d in surviving_synth:
        raw_by_frame_with_synth.setdefault(d.frame, []).append(d)

    store = AnnotationStore(raw_by_frame=raw_by_frame_with_synth, seed=combined)
    gt_path = OUT_DIR / f"{VIDEO_STEM}.gt.csv"
    n_rows = store.export_long_csv(str(gt_path))
    print(f"wrote gt.csv         : {gt_path}  ({n_rows} rows)")

    summary_path = OUT_DIR / f"{VIDEO_STEM}.gt_summary.txt"
    write_export_summary(
        summary_path,
        annotations=combined,
        raw_by_frame=raw_by_frame_with_synth,
        video_props=video_props,
        export_csv_name=gt_path.name,
    )
    print(f"wrote gt_summary     : {summary_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
