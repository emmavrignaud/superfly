"""Export annotated GT detections to a Roboflow-ready zip.

Usage (from repo root):
    python src/export_to_roboflow.py \\
        --gt   data/manual_labelling/<stem>/<stem>.gt.csv \\
        --video data/manual_labelling/<stem>/<stem>.mp4 \\
        --out  roboflow_export.zip \\
        [--class fly]

The zip contains:
    images/frame_XXXXXX.jpg   — one JPG per annotated frame
    annotations.csv           — Roboflow CSV format:
                                filename,width,height,class,xmin,ymin,xmax,ymax

Upload in Roboflow UI:
    Dataset → Add Images → "Upload from Computer" → drag the zip.
    When asked for annotation format, choose "CSV".
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export GT CSV + video frames to Roboflow zip")
    p.add_argument("--gt",    required=True, help="path to .gt.csv (must have x1,y1,x2,y2 columns)")
    p.add_argument("--video", required=True, help="path to the source video file")
    p.add_argument("--out",   required=True, help="output zip file path")
    p.add_argument("--class", dest="fly_class", default="fly",
                   help="class label to use for all detections (default: fly)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gt_path = Path(args.gt)
    video_path = Path(args.video)
    out_path = Path(args.out)

    df = pd.read_csv(gt_path)
    required = {"frame", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"GT CSV is missing columns: {missing}.\n"
            "Re-export from the labeler (Ctrl+E) to get bounding box columns."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_frames = sorted(df["frame"].unique())
    print(f"GT rows: {len(df)}  |  unique frames: {len(annotated_frames)}  |  video: {width}x{height}")

    # Build Roboflow CSV rows
    rf_rows = []
    for _, row in df.iterrows():
        img_name = f"frame_{int(row['frame']):06d}.jpg"
        rf_rows.append({
            "filename": img_name,
            "width":    width,
            "height":   height,
            "class":    args.fly_class,
            "xmin":     row["x1"],
            "ymin":     row["y1"],
            "xmax":     row["x2"],
            "ymax":     row["y2"],
        })
    rf_df = pd.DataFrame(rf_rows, columns=["filename", "width", "height", "class",
                                           "xmin", "ymin", "xmax", "ymax"])

    # Extract frames and write zip
    frame_set = set(annotated_frames)
    frames_written = 0
    current_frame = 0

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write annotation CSV
        csv_buf = io.StringIO()
        rf_df.to_csv(csv_buf, index=False)
        zf.writestr("annotations.csv", csv_buf.getvalue())

        while True:
            ret, frame_img = cap.read()
            if not ret:
                break
            if current_frame in frame_set:
                img_name = f"frame_{current_frame:06d}.jpg"
                ok, buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    zf.writestr(f"images/{img_name}", buf.tobytes())
                    frames_written += 1
            current_frame += 1

    cap.release()
    print(f"Written {frames_written} frames + annotations.csv -> {out_path}")
    print(f"\nUpload instructions:")
    print(f"  1. Go to your Roboflow project -> Add Images")
    print(f"  2. Upload {out_path.name}")
    print(f"  3. When prompted for annotation format, select 'CSV'")


if __name__ == "__main__":
    main()
