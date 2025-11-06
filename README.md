# iabot 计算机视觉脚本说明

- 仓库包含 3 个脚本：
  - `face_detect_dnn.py`：基于 OpenCV DNN 的人脸检测（ResNet-SSD）。支持图片/视频/摄像头；可选五官关键点（Facemark LBF 或 MediaPipe FaceMesh）。
  - `face_detect.py`：基于 OpenCV Haar Cascade 的人脸检测（Viola-Jones）。轻量、依赖少，适合入门与快速测试。
  - `target-locate-1.py`：目标定位（方形检测）。实时摄像头，检测视野中的方形并显示其中心坐标。

## 环境要求

- Python 3.9+（推荐 3.10/3.11）
- Windows 10/11（已在此环境测试）；Linux/macOS 也可运行（注意摄像头后端差异）
- 可选摄像头（用于实时）
- CPU 即可运行；不需要 GPU

## 安装依赖

最低依赖（运行 Haar 与方形检测）：

- 安装：
  - `pip install opencv-python numpy`

人脸 DNN 的关键点可选两种后端，二选一：

- 方案 A（推荐，侧脸更准）：DNN + MediaPipe FaceMesh
  - 安装：`pip install mediapipe opencv-python`
  - 验证：
    - `python -c "import cv2, mediapipe as mp; print(cv2.__version__); print('mediapipe OK')"`

- 方案 B（不安装 mediapipe 时使用）：DNN + OpenCV Facemark LBF
  - 注意：Facemark 属于 OpenCV 的 contrib 模块，需要 `opencv-contrib-python`
  - 安装：
    - 先清理：`pip uninstall -y opencv-python opencv-contrib-python`
    - 再安装：`pip install opencv-contrib-python==4.10.0.84 numpy`
  - 验证：
    - `python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"`

提示：不要同时安装 `opencv-python` 与 `opencv-contrib-python`，以免冲突。根据所需功能选择其一即可（FaceMesh 用 `opencv-python`；Facemark 用 `opencv-contrib-python`）。

## 模型文件（DNN 与 LBF）

- `face_detect_dnn.py` 启动时会自动下载所需模型到本地：
  - DNN 检测：`computer_vision/models/face_dnn/`
    - `deploy.prototxt`
    - `res10_300x300_ssd_iter_140000.caffemodel`
  - LBF 关键点：`computer_vision/models/landmarks/`
    - `lbfmodel.yaml`
  - 若自动下载失败，可手动下载并通过脚本参数指定：
    - `--model-prototxt` 指向 `deploy.prototxt`
    - `--model-weights` 指向 `res10_300x300_ssd_iter_140000.caffemodel`
    - `--landmarks-model` 指向 `lbfmodel.yaml`

## 如何运行

### 1) DNN 人脸检测（face_detect_dnn.py）

- 摄像头实时显示：
  - 基本检测：
    - `python face_detect_dnn.py --source webcam --display`
  - 启用五官标注（推荐 FaceMesh）：
    - `python face_detect_dnn.py --source webcam --display --landmarks --landmarks-method facemesh --draw-features`
  - 启用五官标注（Facemark LBF）：
    - `python face_detect_dnn.py --source webcam --display --landmarks --draw-features`

- 处理图片并保存：
  - `python face_detect_dnn.py --source path/to/your.jpg --display --save`

- 保存标注视频（MP4）或帧/人脸：
  - 标注视频：`python face_detect_dnn.py --source webcam --save --output results/face_dnn.mp4`
  - 标注帧 JPG：`python face_detect_dnn.py --source webcam --save-frames --frame-interval 10`
  - 裁剪人脸 JPG：`python face_detect_dnn.py --source webcam --save-faces`

- 常用参数：
  - 置信阈值：`--conf-threshold 0.6`
  - 分辨率：`--width 1280 --height 720`
  - 摄像头索引：`--camera-index 0`

### 2) Haar 人脸检测（face_detect.py）

- 摄像头实时：`python face_detect.py --source webcam --display`
- 处理图片：`python face_detect.py --source path/to/your.jpg --display --save`
- 参数调优：
  - `--scale-factor 1.2`（检测金字塔步长，1.1–1.3 常用）
  - `--min-neighbors 5`（抑制重叠检测）
  - `--min-size 30`（最小人脸尺寸）
- 可保存标注帧/人脸，与 DNN 类似：`--save-frames --save-faces --frame-interval 10`

### 3) 方形检测（target-locate-1.py）

- 运行：`python target-locate-1.py`
- 功能：实时检测近似正方形，绘制轮廓与中心点，控制台打印坐标；检测到方形自动保存标注帧到 `./results/`。
- 可在脚本中配置：
  - `camera_index`（默认 0）
  - `min_area`（过滤小轮廓）
  - `aspect_tolerance`（宽高比容差，默认 ±0.25）
  - `debug=True` 显示中间结果（阈值、边缘图等）

## 输出目录结构

- `results/face_dnn_*.mp4`：DNN 标注视频输出
- `results/face_dnn_*.jpg`：DNN 标注图片输出
- `results/frames/frames_dnn_*/*.jpg`：DNN 标注帧输出
- `results/faces/faces_dnn_*/*.jpg`：DNN 裁剪人脸输出
- `results/square_*.jpg`：方形检测保存帧

## 核心理论简述

### DNN 人脸检测（ResNet-SSD）

- 模型：Caffe 版 ResNet-SSD，输入 `300x300`，BGR，减均值 `(104, 177, 123)`
- 机制：SSD 为单次前向检测器，通过预定义锚框在不同特征层回归人脸边界与置信度
- 输出：形如 `[1, 1, N, 7]`，包含 `confidence` 与归一化的 `x1,y1,x2,y2`，根据原始 `w,h` 映射到像素坐标
- 优势：速度快、对光照/噪声鲁棒；不足：极端遮挡/超大姿态下检测框可能不稳定

### 五官关键点（两种后端）

- Facemark LBF（68 点）：
  - 原理：局部二值特征 + 级联回归，基于人脸 ROI 在灰度图上迭代拟合形状
  - 特性：对正脸较稳，侧脸/大姿态时可能漂移；本仓库已对人脸框自动扩展（约 1.3 倍）以提升鲁棒性
  - 依赖：`opencv-contrib-python`

- MediaPipe FaceMesh（468/478 点）：
  - 原理：端到端神经网络预测稠密 3D 面部网格点；`refine_landmarks=True` 可输出虹膜点（478）
  - 特性：多视角更稳，侧脸/部分遮挡表现优于传统 LBF；CPU 即可运行
  - 依赖：`mediapipe` + `opencv-python`

### Haar 人脸检测（Viola–Jones）

- 原理：积分图 + Haar-like 特征 + AdaBoost 级联分类器；滑窗与金字塔用于尺度适配
- 优势：实现简单，速度快；劣势：对侧脸/强遮挡鲁棒性较差，误检/漏检在复杂场景更明显

### 方形检测（target-locate-1.py）

- 管线 A：自适应阈值 + 形态学开闭运算 + 轮廓筛选 + 旋转最小外接矩形（`minAreaRect`）
- 管线 B：Canny 边缘 + 形态学 + 轮廓筛选（作为补充路径）
- 判定：近似四边形、凸性；边长近似相等；角度近似 90°；或外接旋转矩形宽高比近似 1

## 常见问题与排查

- 模型下载失败（404/网络问题）：
  - 脚本已内置多个备选链接；仍失败时可手动下载并使用 `--model-*` 参数指定本地路径

- `FacemarkLBF not available`：
  - 安装 `opencv-contrib-python`（见上文方案 B）或改用 `--landmarks-method facemesh`

- `mediapipe` 未安装/导入失败：
  - 安装 `mediapipe`（见方案 A）；在无 GPU 的普通电脑也可运行

- 摄像头无法打开：
  - 切换索引 `--camera-index 0/1/...`；确认其他程序未占用摄像头；Windows 可尝试 `CAP_DSHOW`（在另一个脚本已做兼容）

- 速度问题：
  - 降低分辨率 `--width/--height`；减少标注开销（关闭 `--draw-features`）；帧间隔保存 `--frame-interval`

## 进一步计划（可选）

- 将扩框比例做成参数（例如 `--lbf-expand 1.2`）以适应不同场景
- 为 FaceMesh 提供关闭 `refine_landmarks` 的选项以提速

如需将 Haar 与 DNN 脚本统一成一个入口并可选择模型，请告知，我可以补充一个统一 CLI。