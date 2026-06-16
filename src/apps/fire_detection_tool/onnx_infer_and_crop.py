import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import yaml
from pathlib import Path

# 置信度与IOU阈值
confidence_thres = 0.35
iou_thres = 0.5

# 从 dataset.yaml 读取类别映射
DATASET_YAML_PATH = Path(__file__).parent / "dataset.yaml"

with open(DATASET_YAML_PATH, 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)
classes = dataset_config['names']
print(f"从 dataset.yaml 加载了 {len(classes)} 个类别:")
for i in range(min(5, len(classes))):
    print(f"  {i}: {classes[i]}")
print(f"  ... (共 {len(classes)} 类)\n")

color_palette = np.random.uniform(100, 255, size=(len(classes), 3))

# 动态选择 GPU/CPU provider
def get_providers():
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        print("检测到 onnxruntime-gpu，使用 CUDAExecutionProvider")
        return [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    else:
        print("未检测到 onnxruntime-gpu，使用 CPUExecutionProvider")
        return ['CPUExecutionProvider']

def calculate_iou(box, other_boxes):
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = box[2] * box[3]
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    iou = intersection_area / (box_area + other_boxes_area - intersection_area + 1e-6)
    return iou

def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    if len(boxes) == 0:
        return []
    scores = np.array(scores)
    boxes = np.array(boxes)
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    if len(filtered_boxes) == 0:
        return []
    sorted_indices = np.argsort(filtered_scores)[::-1]
    indices = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        indices.append(current_index)
        if len(sorted_indices) == 1:
            break
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        iou = calculate_iou(current_box, other_boxes)
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    return indices

def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box
    color = color_palette[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    label = f'{classes[class_id]}: {score:.2f}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def preprocess(img, input_width, input_height):
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # CHW
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, img_height, img_width

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _standardize_outputs(raw_output):
    """
    标准化 ONNX 输出为 [num_preds, channels]
    兼容两种形状：(1, C, N) 或 (1, N, C)
    """
    arr = np.squeeze(raw_output[0])
    if arr.ndim == 2:
        h, w = arr.shape
        if h < w:
            return arr.T  # (N, C)
        else:
            return arr    # (N, C)
    elif arr.ndim == 3:
        if arr.shape[1] < arr.shape[2]:
            return np.swapaxes(arr, 1, 2)[0]
        else:
            return arr[0]
    else:
        raise RuntimeError(f"不支持的输出形状: {arr.shape}")

def crop_title_region(original_img, box, output_dir, img_filename, crop_index):
    """
    根据 title 检测框裁剪原始图像（未标注）
    裁剪范围：从 (left - width) 到 right，上下按照 title 的高度
    
    Args:
        original_img: 原始未处理图像
        box: [left, top, width, height]
        output_dir: 输出文件夹路径
        img_filename: 原始图像文件名
        crop_index: 同一张图片中的第几个 title 裁剪
    
    Returns:
        str: 裁剪图像的保存路径
    """
    left, top, width, height = box
    img_height, img_width = original_img.shape[:2]
    
    # 计算裁剪范围
    # 从距离 title 左端 width 距离处开始到 title 右端
    crop_left = max(0, left - width)
    crop_right = min(img_width, left + width)
    crop_top = max(0, top)
    crop_bottom = min(img_height, top + height)
    
    # 从原始图像中裁剪
    cropped_img = original_img[crop_top:crop_bottom, crop_left:crop_right]
    
    # 保存裁剪后的图像
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    stem = Path(img_filename).stem
    suffix = Path(img_filename).suffix
    if crop_index == 0:
        output_filename = f"{stem}_title{suffix}"
    else:
        output_filename = f"{stem}_title_{crop_index}{suffix}"
    output_path = output_dir / output_filename
    
    cv2.imwrite(str(output_path), cropped_img)
    print(f"  ✓ 已裁剪 title 区域: {output_filename} (大小: {cropped_img.shape[1]}x{cropped_img.shape[0]})")
    
    return str(output_path)

def postprocess(input_image, original_image, output, input_width, input_height, img_width, img_height,
                crop_output_dir=None, img_filename=None):
    """
    后处理：解析检测结果、绘制框、裁剪 title
    
    Returns:
        tuple: (output_image, fire_count, title_crop_paths)
    """
    outputs = _standardize_outputs(output)  # [num_preds, channels]
    rows, channels = outputs.shape
    num_classes = channels - 4

    boxes, scores, class_ids = [], [], []
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for i in range(rows):
        row = outputs[i]
        cls_scores = row[4:4 + num_classes]
        max_conf = float(np.max(cls_scores))
        
        if max_conf >= confidence_thres:
            class_id = int(np.argmax(cls_scores))
            # 防止类别越界
            if class_id >= len(classes):
                continue
            
            x, y, w, h = row[0], row[1], row[2], row[3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            # 过滤极小框和越界框
            if width <= 2 or height <= 2:
                continue
            if left < 0 or top < 0 or left + width > img_width or top + height > img_height:
                continue

            class_ids.append(class_id)
            scores.append(max_conf)
            boxes.append([left, top, width, height])

    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    
    # 记录裁剪的 title 个数和 fire 个数
    title_crop_count = 0
    fire_count = 0
    title_crop_paths = []
    
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        draw_detections(input_image, box, score, class_id)
        
        if class_id == 0:  # fire
            fire_count += 1
        elif class_id == 1 and crop_output_dir and img_filename:  # title
            crop_path = crop_title_region(original_image, box, crop_output_dir, img_filename, title_crop_count)
            title_crop_paths.append(crop_path)
            title_crop_count += 1
    
    return input_image, fire_count, title_crop_paths

def init_detect_model(model_path):
    session = ort.InferenceSession(model_path, providers=get_providers())
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]
    return session, model_inputs, input_width, input_height

def detect_object(image, session, model_inputs, input_width, input_height,
                  crop_output_dir=None, img_filename=None):
    """原有的检测函数（保持兼容性）"""
    if isinstance(image, Image.Image):
        result_image = np.array(image)
    else:
        result_image = image.copy()
    
    # 保存原始图像备份
    original_image = result_image.copy()
    
    img_data, img_height, img_width = preprocess(result_image, input_width, input_height)
    outputs = session.run(None, {model_inputs[0].name: img_data})
    output_image, fire_count, title_crop_paths = postprocess(
        result_image, original_image, outputs, 
        input_width, input_height, img_width, img_height,
        crop_output_dir=crop_output_dir, 
        img_filename=img_filename
    )
    return output_image

def detect_single_image(image_path, session, model_inputs, input_width, input_height, crop_output_dir):
    """
    检测单张图像并返回结果（新增的可调用接口）
    
    Args:
        image_path: 图像路径
        session: ONNX 会话
        model_inputs: 模型输入
        input_width: 输入宽度
        input_height: 输入高度
        crop_output_dir: title 裁剪输出目录
    
    Returns:
        tuple: (fire_count, title_crop_paths)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    original_image = img.copy()
    result_image = img.copy()
    
    img_data, img_height, img_width = preprocess(result_image, input_width, input_height)
    outputs = session.run(None, {model_inputs[0].name: img_data})
    
    img_filename = Path(image_path).name
    output_image, fire_count, title_crop_paths = postprocess(
        result_image, original_image, outputs, 
        input_width, input_height, img_width, img_height,
        crop_output_dir=crop_output_dir, 
        img_filename=img_filename
    )
    
    return fire_count, title_crop_paths

def infer_folder(input_dir, output_dir, session, model_inputs, input_width, input_height,
                 crop_output_dir=None):
    """
    批量推理指定文件夹中的图片，并把可视化结果保存到 output_dir。
    检测到 title 时，将裁剪原始图像区域保存到 crop_output_dir。
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 去重收集图片
    files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']:
        for p in in_dir.glob(ext):
            files.add(p.resolve())
    images = sorted(list(files))
    print(f"找到 {len(images)} 张图片\n输出目录: {out_dir.resolve()}")
    if crop_output_dir:
        print(f"Title 裁剪输出目录（原始图像）: {Path(crop_output_dir).resolve()}\n")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] 推理: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  跳过：无法读取 {img_path}")
            continue
        result_image = detect_object(img, session, model_inputs, input_width, input_height,
                                     crop_output_dir=crop_output_dir, img_filename=img_path.name)
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), result_image)
    
    print(f"\n✓ 推理完成！")

if __name__ == '__main__':
    # 模型相对路径
    model_path = str(Path(__file__).parent / "weights" / "best.onnx")

    session, model_inputs, input_width, input_height = init_detect_model(model_path)

    # 模式：1 图片；2 摄像头；3 视频；4 文件夹批量
    mode = 4

    if mode == 1:
        image_data = cv2.imread("street.jpg")
        result_image = detect_object(image_data, session, model_inputs, input_width, input_height)
        cv2.imwrite("output_image.jpg", result_image)
        cv2.imshow('Output', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == 2:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Video", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif mode == 3:
        input_video_path = 'kun.mp4'
        output_video_path = 'kun_det.mp4'
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Info: End of video file.")
                break
            output_image = detect_object(frame, session, model_inputs, input_width, input_height)
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_show = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(output_image, f"FPS: {fps_show:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(output_image)
            cv2.imshow("Output Video", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == 4:
        # 批量推理输入与输出目录
        BASE_DIR = Path(__file__).parent
        INPUT_DIR = str(BASE_DIR / "captured_images")
        OUTPUT_DIR = str(BASE_DIR / "onnx_results")
        CROP_OUTPUT_DIR = str(BASE_DIR / "title_crops")     # title 裁剪输出目录
        
        infer_folder(INPUT_DIR, OUTPUT_DIR, session, model_inputs, input_width, input_height,
                     crop_output_dir=CROP_OUTPUT_DIR)

    else:
        print("输入错误,请检查 mode 的赋值")