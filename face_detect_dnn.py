import argparse
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detection using OpenCV DNN (ResNet-SSD). Supports image, video, and webcam sources."
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
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (0-1)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display window with detections",
    )
    parser.add_argument(
        "--draw-score",
        action="store_true",
        help="Draw detection confidence score on boxes",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to the results folder (image->JPG, video->MP4)",
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
        "--landmarks",
        action="store_true",
        help="Enable facial landmarks (five-feature) detection and annotation",
    )
    parser.add_argument(
        "--landmarks-method",
        type=str,
        default="facemark",
        choices=["facemark", "facemesh"],
        help="Landmarks backend: OpenCV Facemark LBF or MediaPipe FaceMesh",
    )
    parser.add_argument(
        "--landmarks-model",
        type=str,
        default=None,
        help="Path to LBF landmarks model (lbfmodel.yaml). Auto-download if not provided",
    )
    parser.add_argument(
        "--draw-features",
        action="store_true",
        help="Draw five facial features labels (eyes, eyebrows, nose tip, mouth)",
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
    parser.add_argument(
        "--model-prototxt",
        type=str,
        default=None,
        help="Path to deploy.prototxt (optional; auto-download if not provided)",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
        help="Path to res10_300x300_ssd_iter_140000_fp16.caffemodel (auto-download if not provided)",
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


def _download_with_fallback(urls: list[str], dest: Path) -> None:
    last_err = None
    for url in urls:
        try:
            print(f"Downloading from {url} -> {dest}")
            urllib.request.urlretrieve(url, str(dest))
            return
        except Exception as e:
            last_err = e
            print(f"Failed: {e}")
    raise RuntimeError(f"Failed to download model to {dest}. Tried URLs: {urls}. Last error: {last_err}")


def ensure_model_files(args) -> tuple[Path, Path]:
    # Prefer user-specified paths
    if args.model_prototxt and args.model_weights:
        prototxt = Path(args.model_prototxt)
        weights = Path(args.model_weights)
        if not prototxt.exists():
            raise FileNotFoundError(f"Model prototxt not found: {prototxt}")
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        return prototxt, weights

    # Default model directory
    model_dir = Path("computer_vision") / "models" / "face_dnn"
    model_dir.mkdir(parents=True, exist_ok=True)
    prototxt = model_dir / "deploy.prototxt"
    # Use non-FP16 model for broader compatibility
    weights = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    # URLs from OpenCV repositories
    proto_urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
        # Alternative prototxt name used in some samples
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/opencv_face_detector.prototxt",
    ]
    weight_urls = [
        # Recommended by OpenCV samples models.yml (stable tag)
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        # Legacy paths that may still work depending on branch
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        "https://github.com/opencv/opencv_3rdparty/raw/master/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    ]

    # Download if missing
    if not prototxt.exists():
        print(f"Prototxt not found, preparing download to {prototxt} ...")
        _download_with_fallback(proto_urls, prototxt)
    if not weights.exists():
        print(f"Weights not found, preparing download to {weights} ...")
        _download_with_fallback(weight_urls, weights)

    return prototxt, weights


def ensure_landmarks_model(args) -> Path:
    if args.landmarks_model:
        model_path = Path(args.landmarks_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Landmarks model not found: {model_path}")
        return model_path

    model_dir = Path("computer_vision") / "models" / "landmarks"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lbfmodel.yaml"
    if model_path.exists():
        return model_path

    urls = [
        # Widely used LBF model
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
        # OpenCV contrib repo copy
        "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/face/data/lbfmodel.yaml",
    ]
    print(f"Landmarks model not found. Preparing download to {model_path} ...")
    _download_with_fallback(urls, model_path)
    return model_path


def load_facemark(model_path: Path):
    # Requires opencv-contrib-python
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        print(
            "[warn] FacemarkLBF not available. Install 'opencv-contrib-python' to enable landmarks. "
            "Continuing without landmarks."
        )
        return None
    try:
        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel(str(model_path))
        return facemark
    except Exception as e:
        print(f"[warn] Failed to load FacemarkLBF model: {e}. Continuing without landmarks.")
        return None


def load_facemesh():
    try:
        import mediapipe as mp
    except Exception as e:
        print(f"[warn] MediaPipe not available ({e}). Install 'mediapipe' to enable FaceMesh.")
        return None
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return face_mesh


def load_net(prototxt: Path, weights: Path) -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
    return net


def detect_faces_dnn(frame, net: cv2.dnn_Net, conf_threshold: float):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    if detections is None:
        return faces
    # detections: [1, 1, N, 7] => [img_id, class_id, conf, x1, y1, x2, y2]
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
    return faces


def annotate_frame(frame, faces, draw_score: bool):
    for (x, y, w, h, conf) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if draw_score:
            label = f"{conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
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


def _draw_landmarks(frame, landmarks_list):
    for lm in landmarks_list:
        arr = np.asarray(lm)
        # Squeeze singleton dimensions e.g. (1, 68, 2) or (68, 1, 2)
        arr = np.squeeze(arr)
        # If flattened to 1D, reshape back to Nx2
        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        # If still 3D, collapse the middle dim
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arr = arr.reshape(arr.shape[0], 2)
        for (x, y) in arr:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
    return frame


def _feature_centers_from_68(lm):
    # lm: ndarray (68, 2)
    arr = np.asarray(lm)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr.reshape(arr.shape[0], 2)

    def mean_xy(idxs):
        pts = arr[idxs]
        return tuple(np.mean(pts, axis=0).astype(int))

    features = {
        "right_eye": mean_xy(list(range(36, 42))),
        "left_eye": mean_xy(list(range(42, 48))),
        "right_eyebrow": mean_xy(list(range(17, 22))),
        "left_eyebrow": mean_xy(list(range(22, 27))),
        "nose_tip": tuple(arr[30].astype(int)),
        "mouth": mean_xy(list(range(48, 60))),
    }
    return features


def _normalize_landmarks_list(landmarks):
    norm = []
    if not landmarks:
        return norm
    for lm in landmarks:
        arr = np.asarray(lm)
        arr = np.squeeze(arr)
        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arr = arr.reshape(arr.shape[0], 2)
        norm.append(arr)
    return norm


def _expand_rects(faces, frame_shape, scale: float = 1.3):
    h, w = frame_shape[:2]
    expanded = []
    for (x, y, bw, bh, _conf) in faces:
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        nw = bw * scale
        nh = bh * scale
        nx = int(max(0, cx - nw / 2.0))
        ny = int(max(0, cy - nh / 2.0))
        nx2 = int(min(w - 1, nx + nw))
        ny2 = int(min(h - 1, ny + nh))
        expanded.append((nx, ny, max(1, nx2 - nx), max(1, ny2 - ny)))
    return expanded


def detect_landmarks_facemesh(frame, facemesh):
    if facemesh is None:
        return []
    try:
        import mediapipe as mp  # noqa: F401
    except Exception:
        return []
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(rgb)
    lm_list = []
    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            pts = []
            for lm in fl.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))
            lm_list.append(np.array(pts, dtype=np.int32))
    return lm_list


def _feature_centers(lm):
    arr = np.asarray(lm)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    n = arr.shape[0]
    if n >= 468:
        left_eye_idx = [33, 133, 160, 159, 158, 157, 173, 246]
        right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 466]
        left_brow_idx = [70, 63, 105, 66, 107]
        right_brow_idx = [336, 296, 334, 293, 300]
        nose_tip_idx = [1]
        mouth_center_idx = [13, 14]
        def mean_idx(idxs):
            valid = [i for i in idxs if i < n]
            if not valid:
                return (int(arr[:,0].mean()), int(arr[:,1].mean()))
            pts = arr[valid]
            return tuple(np.mean(pts, axis=0).astype(int))
        return {
            "right_eye": mean_idx(right_eye_idx),
            "left_eye": mean_idx(left_eye_idx),
            "right_eyebrow": mean_idx(right_brow_idx),
            "left_eyebrow": mean_idx(left_brow_idx),
            "nose_tip": mean_idx(nose_tip_idx),
            "mouth": mean_idx(mouth_center_idx),
        }
    else:
        return _feature_centers_from_68(arr)


def _draw_feature_labels(frame, landmarks_list):
    for lm in landmarks_list:
        feats = _feature_centers(lm)
        for name, (x, y) in feats.items():
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            cv2.putText(frame, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_image(path: str, net: cv2.dnn_Net, args, facemark=None, facemesh=None):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    start = time.time()
    faces = detect_faces_dnn(img, net, args.conf_threshold)
    annotated = annotate_frame(img.copy(), faces, args.draw_score)
    # Landmarks
    if args.landmarks and faces:
        if args.landmarks_method == "facemesh" and facemesh is not None:
            lm_list = detect_landmarks_facemesh(img, facemesh)
            annotated = _draw_landmarks(annotated, lm_list)
            if args.draw_features:
                annotated = _draw_feature_labels(annotated, lm_list)
        elif facemark is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects_exp = _expand_rects(faces, gray.shape, scale=1.3)
            rects = np.array(rects_exp, dtype=np.int32)
            ok, landmarks = facemark.fit(gray, rects)
            lm_list = _normalize_landmarks_list(landmarks) if ok else []
            annotated = _draw_landmarks(annotated, lm_list)
            if args.draw_features:
                annotated = _draw_feature_labels(annotated, lm_list)
    elapsed = (time.time() - start) * 1000.0
    print(f"Detected {len(faces)} face(s) in {elapsed:.1f} ms")

    if args.display:
        cv2.imshow("Face Detection (DNN) - Image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save:
        ensure_results_dir()
        out_path = (
            Path(args.output)
            if args.output
            else Path("results") / f"face_dnn_{timestamp_str()}.jpg"
        )
        cv2.imwrite(
            str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))]
        )
        print(f"Saved annotated image to {out_path}")


def process_video(source: str, net: cv2.dnn_Net, args, facemark=None):
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
            out_path = Path("results") / f"face_dnn_{timestamp_str()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Prepare directories for frame/face JPG saving
    if args.save_frames:
        run_dir = ensure_subdir("frames") / f"frames_dnn_{timestamp_str()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = run_dir
        print(f"Saving annotated frames to {frames_dir}")
    if args.save_faces:
        run_dir = ensure_subdir("faces") / f"faces_dnn_{timestamp_str()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        faces_dir = run_dir
        print(f"Saving cropped faces to {faces_dir}")

    print("Press 'q' to quit.")
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detect_faces_dnn(frame, net, args.conf_threshold)
        annotated = annotate_frame(frame.copy(), faces, args.draw_score)
        # Landmarks per frame
        if args.landmarks and faces:
            if hasattr(args, "landmarks_method") and args.landmarks_method == "facemesh" and hasattr(process_video, "_facemesh_obj") and process_video._facemesh_obj is not None:
                lm_list = detect_landmarks_facemesh(frame, process_video._facemesh_obj)
                annotated = _draw_landmarks(annotated, lm_list)
                if args.draw_features:
                    annotated = _draw_feature_labels(annotated, lm_list)
            elif facemark is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects_exp = _expand_rects(faces, gray.shape, scale=1.3)
                rects = np.array(rects_exp, dtype=np.int32)
                ok, landmarks = facemark.fit(gray, rects)
                lm_list = _normalize_landmarks_list(landmarks) if ok else []
                annotated = _draw_landmarks(annotated, lm_list)
                if args.draw_features:
                    annotated = _draw_feature_labels(annotated, lm_list)

        if args.display:
            cv2.imshow("Face Detection (DNN) - Video", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if writer is not None:
            writer.write(annotated)

        # Save annotated frames as JPG
        if frames_dir is not None and (frame_count % max(1, args.frame_interval) == 0):
            jpg_path = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(
                str(jpg_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))]
            )

        # Save cropped faces as JPG
        if faces_dir is not None and faces:
            for idx, (x, y, w, h, conf) in enumerate(faces):
                roi = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
                if roi.size == 0:
                    continue
                face_path = faces_dir / f"frame_{frame_count:06d}_face_{idx:02d}.jpg"
                cv2.imwrite(
                    str(face_path), roi, [cv2.IMWRITE_JPEG_QUALITY, int(max(0, min(args.jpg_quality, 100)))]
                )

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
    prototxt, weights = ensure_model_files(args)
    net = load_net(prototxt, weights)
    # Prepare landmarks model if enabled
    facemark = None
    facemesh = None
    if args.landmarks:
        if hasattr(args, "landmarks_method") and args.landmarks_method == "facemesh":
            facemesh = load_facemesh()
            if facemesh is None:
                print("[info] FaceMesh not available; falling back to OpenCV Facemark if possible.")
                lm_model_path = ensure_landmarks_model(args)
                facemark = load_facemark(lm_model_path)
        else:
            lm_model_path = ensure_landmarks_model(args)
            facemark = load_facemark(lm_model_path)
        if facemark is None and facemesh is None:
            print("[info] Landmarks disabled: no backend available.")

    # stash facemesh into process_video function attribute to avoid signature changes across calls
    process_video._facemesh_obj = facemesh

    if args.source == "webcam":
        process_video("webcam", net, args, facemark)
        return

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source not found: {args.source}")

    if is_image_file(args.source):
        process_image(args.source, net, args, facemark, facemesh)
    else:
        process_video(args.source, net, args, facemark)


if __name__ == "__main__":
    main()