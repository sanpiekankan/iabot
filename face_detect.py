import argparse
import os
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detection using OpenCV Haar Cascade. Supports image, video, and webcam sources."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Input source: 'webcam' or a path to an image/video file",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index when source is 'webcam'",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for the Haar detector (typically 1.1–1.3)",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Min neighbors for Haar detector (higher reduces false positives)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=30,
        help="Minimum face size in pixels",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display window with detections",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to the results folder",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames as JPG images (video/webcam)",
    )
    parser.add_argument(
        "--save-faces",
        action="store_true",
        help="Save cropped detected faces as JPG images (video/webcam)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Save every N-th frame when using --save-frames",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=90,
        help="JPEG quality (0-100) for saved JPG images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for saving annotated image/video",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture width for webcam/video",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture height for webcam/video",
    )
    return parser.parse_args()


def ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def ensure_subdir(name: str) -> Path:
    base = ensure_results_dir() / name
    base.mkdir(parents=True, exist_ok=True)
    return base


def timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_cascade() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return cascade


def detect_faces(gray, cascade, scale_factor: float, min_neighbors: int, min_size: int):
    return cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def annotate_frame(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    return frame


def process_image(path: str, cascade: cv2.CascadeClassifier, args):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detect_faces(gray, cascade, args.scale_factor, args.min_neighbors, args.min_size)
    annotated = annotate_frame(img.copy(), faces)
    elapsed = (time.time() - start) * 1000.0
    print(f"Detected {len(faces)} face(s) in {elapsed:.1f} ms")

    if args.display:
        cv2.imshow("Face Detection - Image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save:
        ensure_results_dir()
        out_path = (
            Path(args.output)
            if args.output
            else Path("results") / f"face_{timestamp_str()}.jpg"
        )
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))])
        print(f"Saved annotated image to {out_path}")


def process_video(source: str, cascade: cv2.CascadeClassifier, args):
    cap = cv2.VideoCapture(args.camera_index if source == "webcam" else source)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    out_path = None
    frames_dir = None
    faces_dir = None
    if args.save:
        ensure_results_dir()
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path("results") / f"face_{timestamp_str()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Prepare directories for frame/face JPG saving
    if args.save_frames:
        run_dir = ensure_subdir("frames") / f"frames_{timestamp_str()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = run_dir
        print(f"Saving annotated frames to {frames_dir}")
    if args.save_faces:
        run_dir = ensure_subdir("faces") / f"faces_{timestamp_str()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        faces_dir = run_dir
        print(f"Saving cropped faces to {faces_dir}")

    print("Press 'q' to quit.")
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detect_faces(gray, cascade, args.scale_factor, args.min_neighbors, args.min_size)
        annotated = annotate_frame(frame, faces)

        if args.display:
            cv2.imshow("Face Detection - Video", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if writer is not None:
            writer.write(annotated)

        # Save annotated frames as JPG
        if frames_dir is not None and (frame_count % max(1, args.frame_interval) == 0):
            jpg_path = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(jpg_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))])

        # Save cropped faces as JPG
        if faces_dir is not None and faces is not None and len(faces) > 0:
            for idx, (x, y, w, h) in enumerate(faces):
                roi = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                if roi.size == 0:
                    continue
                face_path = faces_dir / f"frame_{frame_count:06d}_face_{idx:02d}.jpg"
                cv2.imwrite(str(face_path), roi, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))])

        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {out_path}")
    if args.display:
        cv2.destroyAllWindows()


def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main():
    args = parse_args()
    cascade = load_cascade()

    if args.source == "webcam":
        process_video("webcam", cascade, args)
        return

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source not found: {args.source}")

    if is_image_file(args.source):
        process_image(args.source, cascade, args)
    else:
        process_video(args.source, cascade, args)


if __name__ == "__main__":
    main()