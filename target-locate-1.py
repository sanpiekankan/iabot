"""
目标定位（方形检测）
基于 Python 与 OpenCV，实现实时打开本地摄像头，检测视野中的方形物体，框选并显示其中心坐标 (x, y)。

使用方法：
1) 安装依赖：
   pip install opencv-python numpy
2) 运行：
   python target-locate-1.py
3) 退出：
   在显示窗口按键 'q' 退出。
"""

import cv2
import numpy as np
import os
from datetime import datetime


class TargetLocator:
    def __init__(
        self,
        camera_index: int = 0,
        min_area: float = 500.0,
        aspect_tolerance: float = 0.25,
        debug: bool = False,
    ) -> None:
        """初始化目标定位器。

        参数:
            camera_index: 摄像头索引（默认 0）。
            min_area: 轮廓最小面积过滤，小于此面积的轮廓忽略。
            aspect_tolerance: 方形宽高比的容差，1±tolerance 被认为是方形。
            debug: 是否显示中间处理结果用于调试。
        """
        self.camera_index = camera_index
        self.min_area = float(min_area)
        self.aspect_tolerance = float(aspect_tolerance)
        self.debug = bool(debug)

        self.cap = None
        self.save_dir = "results"

    def open_camera(self) -> None:
        # Windows 上使用 CAP_DSHOW 可减少警告/提升兼容性
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        except Exception:
            # 兼容其他平台
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头索引 {self.camera_index}")

    @staticmethod
    def _auto_canny_thresholds(gray: np.ndarray) -> tuple[int, int]:
        # 根据灰度图的中位数估计 Canny 阈值
        v = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        # 保证阈值有效范围
        lower = max(5, min(lower, 250))
        upper = max(lower + 10, min(upper, 255))
        return lower, upper

    @staticmethod
    def _is_square(approx: np.ndarray, aspect_tolerance: float) -> bool:
        # 近似多边形是 4 点且凸，进一步检查边长近似相等与角度近似 90°
        if approx is None or len(approx) != 4:
            return False

        pts = approx.reshape(-1, 2).astype(np.float32)

        # 边长（顺时针）
        lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            lengths.append(np.linalg.norm(p2 - p1))

        max_len = max(lengths)
        min_len = min(lengths)
        if min_len <= 1e-3:
            return False

        # 边长比约束（近似相等）
        if max_len / min_len > (1.0 + aspect_tolerance):
            return False

        # 角度约束（近似直角）：余弦绝对值接近 0
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i - 1) % 4]
            p2 = pts[(i + 1) % 4]
            v1 = p1 - p0
            v2 = p2 - p0
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_angle = abs(float(np.dot(v1, v2)) / denom)
            if cos_angle > 0.3:  # 约 72° ~ 108° 之间允许误差
                return False

        return True

    @staticmethod
    def _rect_is_square(rect: tuple, aspect_tolerance: float) -> bool:
        # 使用旋转最小外接矩形判定是否近似正方形（对旋转更鲁棒）
        # rect: ((cx, cy), (w, h), angle)
        (w, h) = rect[1]
        w = float(w)
        h = float(h)
        if w < 5 or h < 5:
            return False
        # 宽高比接近 1 即认为是方形（ratio >= 1）
        ratio = max(w, h) / max(1.0, min(w, h))
        return ratio <= (1.0 + aspect_tolerance)

    def detect_squares(self, frame: np.ndarray) -> list[dict]:
        """在单帧图像中检测方形，返回检测结果列表。"""
        # 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        results = []

        # 路线 A：自适应阈值（对光照和纹理更鲁棒）
        bin_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            if not self._rect_is_square(rect, self.aspect_tolerance):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            (cx, cy) = (int(rect[0][0]), int(rect[0][1]))
            x, y, w, h = cv2.boundingRect(approx)
            results.append({
                "approx": approx,
                "box": box,
                "bbox": (x, y, w, h),
                "center": (cx, cy),
            })

        # 路线 B：边缘 + 轮廓（作为补充）
        if not results:
            lower, upper = self._auto_canny_thresholds(blurred)
            edges = cv2.Canny(blurred, lower, upper)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours2, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours2:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                rect = cv2.minAreaRect(cnt)
                if not self._rect_is_square(rect, self.aspect_tolerance):
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                (cx, cy) = (int(rect[0][0]), int(rect[0][1]))
                x, y, w, h = cv2.boundingRect(approx)
                results.append({
                    "approx": approx,
                    "box": box,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                })

        if self.debug:
            cv2.imshow("binary", bin_mask)

        return results

    @staticmethod
    def annotate_frame(frame: np.ndarray, squares: list[dict]) -> np.ndarray:
        out = frame.copy()
        for s in squares:
            x, y, w, h = s["bbox"]
            cx, cy = s["center"]
            box = s.get("box")
            # 旋转矩形框（更贴合真实方形）
            if box is not None:
                cv2.polylines(out, [box], True, (0, 255, 0), 2)
            else:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 中心点
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            # 显示坐标（中心点）
            label = f"x: {cx}, y: {cy}"
            cv2.putText(out, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return out

    def save_frame(self, image: np.ndarray) -> str:
        """保存带标注的当前帧到 ./results/xxx.jpg 并返回路径。"""
        os.makedirs(self.save_dir, exist_ok=True)
        fname = f"square_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        fpath = os.path.join(self.save_dir, fname)
        cv2.imwrite(fpath, image)
        return fpath

    def run(self) -> None:
        """打开摄像头循环处理帧并显示检测结果。按 'q' 键退出。"""
        self.open_camera()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("从摄像头读取失败")
                    break

                squares = self.detect_squares(frame)
                annotated = self.annotate_frame(frame, squares)

                # 控制台打印每帧检测到的坐标
                for s in squares:
                    cx, cy = s["center"]
                    print(f"Square at x={cx}, y={cy}")

                # 检测到方形则保存当前帧（含绿色轮廓与坐标标注）
                if squares:
                    saved_path = self.save_frame(annotated)
                    print(f"Saved annotated frame: {saved_path}")

                cv2.imshow("Target Locate - Squares", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # 可根据需要修改参数，例如 camera_index、min_area、debug
    locator = TargetLocator(camera_index=0, min_area=500, aspect_tolerance=0.25, debug=False)
    locator.run()