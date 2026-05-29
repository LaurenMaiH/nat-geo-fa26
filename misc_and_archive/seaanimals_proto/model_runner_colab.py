import streamlit as st
import os
import csv
import io
import tempfile
import zipfile
import subprocess
import shutil
from pathlib import Path

import numpy as np
import cv2

st.set_page_config(
    page_title="Sea Animals Video Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Banner — looks for natgeobanner.png next to this file, then in /content/drive
_banner_candidates = [
    Path(__file__).parent / "natgeobanner.png",
    Path("/content/drive/My Drive/natgeobanner.png"),
    Path("/content/natgeobanner.png"),
]
for _p in _banner_candidates:
    if _p.exists():
        st.image(str(_p), use_container_width=True)
        break

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_WIDTH  = 854
TARGET_HEIGHT = 480
CROP_PAD      = 10  # pixels of padding around each bounding-box crop

AQUATIC_CLASSES_DEFAULT = [
    'fish', 'crab', 'shrimp', 'lobster', 'octopus',
    'squid', 'jellyfish', 'sea turtle', 'eel',
    'starfish', 'ray', 'shark',
]

# ── Model loader (cached so it loads only once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO-World model…")
def load_model(classes: tuple):
    from ultralytics import YOLOWorld
    model = YOLOWorld('yolov8x-worldv2.pt')
    model.set_classes(list(classes))
    return model

# ── Color enhancement helpers ────────────────────────────────────────────────
def white_balance(frame: np.ndarray) -> np.ndarray:
    result = frame.copy().astype(np.float32)
    mb, mg, mr = result[:,:,0].mean(), result[:,:,1].mean(), result[:,:,2].mean()
    mean_all = (mb + mg + mr) / 3.0
    if mb > 1:
        result[:,:,0] *= mean_all / mb
    if mg > 1:
        result[:,:,1] *= mean_all / mg
    if mr > 1:
        result[:,:,2] *= mean_all / mr
    return np.clip(result, 0, 255).astype(np.uint8)

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lch, a, b = cv2.split(lab)
    lch = _clahe.apply(lch)
    return cv2.cvtColor(cv2.merge([lch, a, b]), cv2.COLOR_LAB2BGR)

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    return apply_clahe(white_balance(frame))

# ── Video helpers ────────────────────────────────────────────────────────────
def trim_black_leader(input_path: str, output_path: str, threshold=10, max_scan_sec=30) -> str:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(max_scan_sec * fps)
    start_sec = 0.0
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.mean() > threshold:
            start_sec = i / fps
            break
    cap.release()
    cmd = ['ffmpeg', '-y', '-i', input_path,
           '-ss', f'{start_sec:.3f}',
           '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-an', output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

# ── Main tracking function ───────────────────────────────────────────────────
def process_video(
    input_path: str,
    video_name: str,
    apply_clahe_flag: bool,
    trim_black: bool,
    frame_skip: int,
    aquatic_classes: list,
    progress_bar,
    status_text,
    crops_dir: str,
) -> tuple:
    """
    Run YOLO-World + ByteTrack on a video.
    Saves first-detection crops to crops_dir.
    Returns (tracked_video_path, csv_path).
    """
    import supervision as sv
    from supervision import VideoInfo, VideoSink, get_video_frames_generator
    from supervision import BoxAnnotator, LabelAnnotator

    model = load_model(tuple(aquatic_classes))

    work_path = input_path

    if trim_black:
        status_text.text("Trimming black frames…")
        trimmed = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        trim_black_leader(work_path, trimmed)
        work_path = trimmed

    raw_info = VideoInfo.from_video_path(work_path)
    video_info = VideoInfo(
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        fps=raw_info.fps,
        total_frames=raw_info.total_frames,
    )

    W, H = TARGET_WIDTH, TARGET_HEIGHT
    margin = int(min(W, H) * 0.03)
    border_polygon = np.array([
        [margin,     margin    ],
        [W - margin, margin    ],
        [W - margin, H - margin],
        [margin,     H - margin],
    ])

    tracker           = sv.ByteTrack(frame_rate=int(video_info.fps))
    box_annotator     = BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=2)
    label_annotator   = LabelAnnotator(text_scale=0.5, text_thickness=1)
    polygon_zone      = sv.PolygonZone(polygon=border_polygon)
    polygon_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone, color=sv.Color.WHITE,
        thickness=2, text_scale=0.6, opacity=0.08,
    )

    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    csv_path     = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

    total_frames   = raw_info.total_frames
    tracked_frames = (total_frames + frame_skip - 1) // frame_skip

    seen_tracker_ids: set = set()

    with VideoSink(output_video, video_info) as sink, \
         open(csv_path, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["video", "frame", "tracker_id", "class", "confidence",
                         "x1", "y1", "x2", "y2"])

        for iter_idx, frame in enumerate(
            get_video_frames_generator(work_path, stride=frame_skip)
        ):
            frame_idx = iter_idx * frame_skip

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            inference_frame = enhance_frame(frame.copy()) if apply_clahe_flag else frame.copy()

            results    = model(inference_frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            if detections.tracker_id is not None:
                for tid, cid, conf, bbox in zip(
                    detections.tracker_id,
                    detections.class_id,
                    detections.confidence,
                    detections.xyxy,
                ):
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    writer.writerow([
                        video_name, frame_idx, int(tid),
                        aquatic_classes[cid], f"{conf:.4f}",
                        x1, y1, x2, y2,
                    ])

                    # Save first-detection crop for each unique tracker
                    if tid not in seen_tracker_ids:
                        seen_tracker_ids.add(tid)
                        x1p = max(0, x1 - CROP_PAD)
                        y1p = max(0, y1 - CROP_PAD)
                        x2p = min(W, x2 + CROP_PAD)
                        y2p = min(H, y2 + CROP_PAD)
                        crop = frame[y1p:y2p, x1p:x2p]
                        class_slug = aquatic_classes[cid].replace(' ', '_')
                        crop_name  = f"{video_name}__{class_slug}__id{int(tid):04d}.jpg"
                        cv2.imwrite(os.path.join(crops_dir, crop_name), crop)

            labels = [
                f"#{tid} {aquatic_classes[cid]} {conf:.2f}"
                for tid, cid, conf in zip(
                    detections.tracker_id if detections.tracker_id is not None else [],
                    detections.class_id,
                    detections.confidence,
                )
            ]

            polygon_zone.trigger(detections=detections)
            annotated = inference_frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            annotated = polygon_annotator.annotate(scene=annotated)
            sink.write_frame(annotated)

            progress_bar.progress((iter_idx + 1) / tracked_frames)
            if iter_idx % 10 == 0:
                status_text.text(f"Processing frame {frame_idx} / {total_frames}…")

    status_text.text("Re-encoding for browser compatibility…")
    final_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run(
        ['ffmpeg', '-y', '-i', output_video,
         '-vcodec', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', final_video],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.unlink(output_video)

    return final_video, csv_path

# ── Sidebar settings ─────────────────────────────────────────────────────────
st.sidebar.header("Settings")

apply_clahe_flag = st.sidebar.toggle("CLAHE + White Balance", value=True,
    help="Boosts contrast and removes blue-green cast in underwater footage.")

trim_black = st.sidebar.toggle("Trim Black Leader", value=False,
    help="Strip leading black frames before tracking.")

frame_skip = st.sidebar.slider("Frame Skip", min_value=1, max_value=10, value=1,
    help="Process every Nth frame. Higher = faster but less accurate tracking.")

aquatic_classes = st.sidebar.multiselect(
    "Tracked Classes",
    options=AQUATIC_CLASSES_DEFAULT,
    default=AQUATIC_CLASSES_DEFAULT,
)

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("Sea Animals Video Processor")
st.caption("YOLO-World + ByteTrack · T4 GPU via Google Colab · upload one or more videos")

uploaded_files = st.file_uploader(
    "Upload video file(s) (.mov or .mp4)",
    type=["mov", "mp4"],
    accept_multiple_files=True,
)

if uploaded_files:
    if not aquatic_classes:
        st.warning("Select at least one tracked class in the sidebar.")
    elif st.button(f"Process {len(uploaded_files)} Video(s)", type="primary", use_container_width=True):
        crops_dir  = tempfile.mkdtemp()
        all_results = []  # (original_name, tracked_path, csv_path)

        for uploaded_file in uploaded_files:
            video_stem = Path(uploaded_file.name).stem

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                input_path = tmp.name

            st.markdown(f"**Processing: {uploaded_file.name}**")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            try:
                tracked_video, csv_path = process_video(
                    input_path,
                    video_name=video_stem,
                    apply_clahe_flag=apply_clahe_flag,
                    trim_black=trim_black,
                    frame_skip=frame_skip,
                    aquatic_classes=aquatic_classes,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    crops_dir=crops_dir,
                )
            except Exception as e:
                st.error(f"Processing failed for {uploaded_file.name}: {e}")
                continue

            progress_bar.empty()
            status_text.empty()
            all_results.append((uploaded_file.name, tracked_video, csv_path))

        if not all_results:
            st.stop()

        st.success(f"Done! Processed {len(all_results)} video(s).")

        import pandas as pd

        # ── Combine CSVs ──────────────────────────────────────────────────────
        dfs = [pd.read_csv(csv_path) for _, _, csv_path in all_results]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_csv_bytes = combined_df.to_csv(index=False).encode()

        # ── Build tracked-videos ZIP ──────────────────────────────────────────
        videos_zip_buf = io.BytesIO()
        with zipfile.ZipFile(videos_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for orig_name, tracked_path, _ in all_results:
                stem = Path(orig_name).stem
                with open(tracked_path, "rb") as f:
                    zf.writestr(f"{stem}_tracked.mp4", f.read())
        videos_zip_bytes = videos_zip_buf.getvalue()

        # ── Build creature-crops ZIP ──────────────────────────────────────────
        crop_files = os.listdir(crops_dir)
        crops_zip_buf = io.BytesIO()
        with zipfile.ZipFile(crops_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in crop_files:
                zf.write(os.path.join(crops_dir, fname), fname)
        crops_zip_bytes = crops_zip_buf.getvalue()

        # ── Download buttons ──────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Download Tracked Videos (ZIP)" if len(all_results) > 1 else "Download Tracked Video",
                data=videos_zip_bytes if len(all_results) > 1 else open(all_results[0][1], "rb").read(),
                file_name="tracked_videos.zip" if len(all_results) > 1 else "tracked_video.mp4",
                mime="application/zip" if len(all_results) > 1 else "video/mp4",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="Download Bounding Box CSV",
                data=combined_csv_bytes,
                file_name="bounding_boxes.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                label=f"Download Creature Crops ({len(crop_files)} images)",
                data=crops_zip_bytes,
                file_name="creature_crops.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # ── Single-video preview ──────────────────────────────────────────────
        if len(all_results) == 1:
            st.subheader("Tracked Video Preview")
            st.video(all_results[0][1])

        # ── Summary table ─────────────────────────────────────────────────────
        st.subheader("First Detection per Tracked Animal")
        first_seen = (
            combined_df.sort_values("frame")
            .groupby(["video", "tracker_id"]).first()
            .reset_index()
        )
        first_seen["width"]  = first_seen["x2"] - first_seen["x1"]
        first_seen["height"] = first_seen["y2"] - first_seen["y1"]
        display_cols = ["video", "tracker_id", "class", "confidence", "frame",
                        "x1", "y1", "x2", "y2", "width", "height"]
        st.dataframe(
            first_seen[[c for c in display_cols if c in first_seen.columns]],
            use_container_width=True,
        )
        st.caption(f"Total unique animals tracked: **{len(first_seen)}**")

        shutil.rmtree(crops_dir, ignore_errors=True)
