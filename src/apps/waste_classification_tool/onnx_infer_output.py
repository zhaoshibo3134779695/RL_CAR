#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能说明：
    1. 类别分组：以垃圾桶类别为基准创建分组（检测到几个垃圾桶就几个组）
    2. 阈值匹配：将开合状态和垃圾种类通过 X_CENTER_THRESHOLD 分配到对应垃圾桶
    3. 冲突解决：每类取置信度最高的一个
    4. 投放判断：根据 WASTE_TO_BIN_TYPE 映射判断垃圾是否投放正确
    5. 输出格式：
        可视化图片保存到 OUTPUT_DIR
        文本日志保存到 detection_log.txt
        终端实时打印播报消息

可调参数：
    1. X_CENTER_THRESHOLD：中点匹配阈值，用于将状态和垃圾分配到垃圾桶
    2. CONFIDENCE_THRES：默认置信度阈值（未在 CLASS_CONF_THRES 指定的类使用它）
    3. INPUT_DIR/OUTPUT_DIR：输入输出目录
"""
import cv2
import onnxruntime as ort
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict
from runtime_paths import BASE_DIR, CAPTURE_DIR, OUTPUT_DIR, WEIGHTS_DIR, VOICE_RUNTIME_DIR, ensure_dirs

# ========== 配置 ==========
DATASET_YAML_PATH = str(BASE_DIR / "dataset.yaml")
MODEL_PATH = str(WEIGHTS_DIR / "best_m.onnx")
INPUT_DIR = str(CAPTURE_DIR)
OUTPUT_DIR = str(OUTPUT_DIR)

# 阈值配置
CONFIDENCE_THRES = 0.20     # 默认置信度阈值（兜底）
IOU_THRES = 0.5             # NMS IoU阈值
X_CENTER_THRESHOLD = 30     # 中点横坐标差值阈值（像素），用于将状态和垃圾分配到垃圾桶

# 按类别单独阈值（键为 dataset.yaml 里的类名）
CLASS_CONF_THRES = {
    "open": 0.35,
    "close": 0.35,
    "Kitchen Waste": 0.35,
    "Recyclables": 0.35,
    "Other Waste": 0.35,
    "Harmful Waste": 0.35,
}

def get_conf_thres_for_class(class_name: str) -> float:
    return CLASS_CONF_THRES.get(class_name, CONFIDENCE_THRES)

# ========== 类别定义 ==========
# 第一类：开合状态
STATE_CLASSES = {"close", "open"}

# 第二类：垃圾桶类别
BIN_TYPE_CLASSES = {
    "Kitchen Waste": "厨余",
    "Recyclables": "可回收",
    "Other Waste": "其他",
    "Harmful Waste": "有害"
}

# 第三类：垃圾种类（108种）
WASTE_CLASSES = set()

# 垃圾种类，英文映射到中文
WASTE_NAME_CN = {
    # 可回收物
    "Waste computers": "废电脑", "Perfume bottle": "香水瓶", "Old Hat": "旧帽子",
    "Gas tank": "煤气罐", "Zip-top can": "易拉罐", "Glass bottles": "玻璃瓶",
    "Old iron pot": "旧铁锅", "Glass Kettle": "玻璃壶", "Old paper bags": "旧纸袋",
    "Old plastic basket": "旧塑料篮子", "Basketball": "篮球", "Carton": "纸盒",
    "Old toys": "旧玩具", "Milk box": "牛奶盒", "Waste lock": "废锁头",
    "Newspaper": "报纸", "Umbrella skeleton": "雨伞骨架", "Spoon": "勺子",
    "Old shoes": "旧鞋子", "Old schoolbag": "旧书包", "Screw": "螺丝钉",
    "Old dolls": "旧玩偶", "Old clip": "旧夹子", "Book": "书本",
    "Plastic bottles": "塑料瓶", "Bad trash can": "坏垃圾桶",
    
    # 其他垃圾
    "Watercolor pen": "水彩笔", "Chopping board": "菜板", "Muck": "渣土",
    "Tiles": "瓦片", "Toothbrush": "牙刷", "Broom": "扫把",
    "Old Mirror": "旧镜子", "Cigarette end": "烟蒂", "Lighter": "打火机",
    "Plastic bag": "塑料袋", "Glue": "胶水", "Wooden comb": "木质梳子",
    "Earthen jar": "坛子", "Old bathtub": "旧浴缸", "Cosmetic brush": "化妆刷",
    "Old Water tank": "旧水槽", "Bad closestool": "坏马桶", "Dirty clothes": "脏污衣服",
    "Shell": "贝壳", "Ceramic bowl": "陶瓷碗", "Toothpaste tube": "牙膏皮",
    "Cup": "杯子", "Bad flowerpots": "坏的花盆", "Dish": "盘子",
    "Towel": "毛巾", "One-off chopsticks": "一次性筷子", "Toilet paper": "卫生纸",
    "Sponge": "海绵", "Brick": "砖块",
    
    # 厨余垃圾
    "Peas peel": "豌豆皮", "Abandoned tea": "废弃茶叶", "Bone": "骨头",
    "Biscuits": "饼干", "Bread": "面包", "Orange peel": "橙皮",
    "Kernel": "桃核", "Tomatoes": "西红柿", "Sugar cane": "甘蔗",
    "Mushroom": "蘑菇", "Ketchup": "番茄酱", "Banana peel": "香蕉皮",
    "Scallion": "葱", "Pear": "梨", "Garlic": "大蒜",
    "Cake": "蛋糕", "Chocolates": "巧克力", "Shrimp shell": "虾壳",
    "Corn": "玉米", "Chilli": "辣椒", "Egg shell": "鸡蛋壳",
    "Vegetable leaf": "菜叶", "Crab shell": "蟹壳", "Fishbone": "鱼骨",
    "Strawberry": "草莓", "eggplant": "茄子", "Prune kernel": "西梅核",
    "Peanut shell": "花生壳", "Potato": "马铃薯", "Watermelon rind": "西瓜皮",
    "Apple": "苹果",
    
    # 有害垃圾
    "Expired Nail Polish": "过期指甲油", "Medical gauze": "医用纱布", "Paint": "油漆",
    "Palette": "调色板", "Hair gel": "发胶", "Alcohol": "酒精",
    "Oral liquid bottle": "口服液瓶", "Waste thermometer": "废温度计", "Syringe": "注射器",
    "Pesticide": "杀虫剂", "Expired Capsule medicine": "过期的胶囊药物", "Paint bucket": "漆桶",
    "Woundplast": "创可贴", "Pesticide bottle": "农药瓶", "Light stick": "荧光棒",
    "Medical cotton swab": "医用棉签", "Medical gloves": "医用手套", "Expired cosmetics": "过期的化妆品",
    "Expired tablets": "过期药品", "Battery": "蓄电池", "Fluorescent Lamp": "荧光灯",
    "Waste battery": "废旧电池"
}

# 垃圾分类映射
WASTE_TO_BIN_TYPE = {
    # 可回收物
    "Waste computers": "可回收", "Perfume bottle": "可回收", "Old Hat": "可回收",
    "Gas tank": "可回收", "Zip-top can": "可回收", "Glass bottles": "可回收",
    "Old iron pot": "可回收", "Glass Kettle": "可回收", "Old paper bags": "可回收",
    "Old plastic basket": "可回收", "Basketball": "可回收", "Carton": "可回收",
    "Old toys": "可回收", "Milk box": "可回收", "Waste lock": "可回收",
    "Newspaper": "可回收", "Umbrella skeleton": "可回收", "Spoon": "可回收",
    "Old shoes": "可回收", "Old schoolbag": "可回收", "Screw": "可回收",
    "Old dolls": "可回收", "Old clip": "可回收", "Book": "可回收",
    "Plastic bottles": "可回收", "Bad trash can": "可回收",
    
    # 其他垃圾
    "Watercolor pen": "其他", "Chopping board": "其他", "Muck": "其他",
    "Tiles": "其他", "Toothbrush": "其他", "Broom": "其他",
    "Old Mirror": "其他", "Cigarette end": "其他", "Lighter": "其他",
    "Plastic bag": "其他", "Glue": "其他", "Wooden comb": "其他",
    "Earthen jar": "其他", "Old bathtub": "其他", "Cosmetic brush": "其他",
    "Old Water tank": "其他", "Bad closestool": "其他", "Dirty clothes": "其他",
    "Shell": "其他", "Ceramic bowl": "其他", "Toothpaste tube": "其他",
    "Cup": "其他", "Bad flowerpots": "其他", "Dish": "其他",
    "Towel": "其他", "One-off chopsticks": "其他", "Toilet paper": "其他",
    "Sponge": "其他", "Brick": "其他",
    
    # 厨余垃圾
    "Peas peel": "厨余", "Abandoned tea": "厨余", "Bone": "厨余",
    "Biscuits": "厨余", "Bread": "厨余", "Orange peel": "厨余",
    "Kernel": "厨余", "Tomatoes": "厨余", "Sugar cane": "厨余",
    "Mushroom": "厨余", "Ketchup": "厨余", "Banana peel": "厨余",
    "Scallion": "厨余", "Pear": "厨余", "Garlic": "厨余",
    "Cake": "厨余", "Chocolates": "厨余", "Shrimp shell": "厨余",
    "Corn": "厨余", "Chilli": "厨余", "Egg shell": "厨余",
    "Vegetable leaf": "厨余", "Crab shell": "厨余", "Fishbone": "厨余",
    "Strawberry": "厨余", "eggplant": "厨余", "Prune kernel": "厨余",
    "Peanut shell": "厨余", "Potato": "厨余", "Watermelon rind": "厨余",
    "Apple": "厨余",
    
    # 有害垃圾
    "Expired Nail Polish": "有害", "Medical gauze": "有害", "Paint": "有害",
    "Palette": "有害", "Hair gel": "有害", "Alcohol": "有害",
    "Oral liquid bottle": "有害", "Waste thermometer": "有害", "Syringe": "有害",
    "Pesticide": "有害", "Expired Capsule medicine": "有害", "Paint bucket": "有害",
    "Woundplast": "有害", "Pesticide bottle": "有害", "Light stick": "有害",
    "Medical cotton swab": "有害", "Medical gloves": "有害", "Expired cosmetics": "有害",
    "Expired tablets": "有害", "Battery": "有害", "Fluorescent Lamp": "有害",
    "Waste battery": "有害"
}

# ========== 工具函数 ==========
def get_providers():
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
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

def preprocess(img, input_width, input_height):
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, img_height, img_width

def _standardize_outputs(raw_output):
    arr = np.squeeze(raw_output[0])
    if arr.ndim == 2:
        h, w = arr.shape
        if h < w:
            return arr.T
        else:
            return arr
    elif arr.ndim == 3:
        if arr.shape[1] < arr.shape[2]:
            return np.swapaxes(arr, 1, 2)[0]
        else:
            return arr[0]
    else:
        raise RuntimeError(f"不支持的输出形状: {arr.shape}")

def postprocess_detections(output, input_width, input_height, img_width, img_height, classes):
    """
    后处理：返回检测结果列表
    每个元素为 {class_name, confidence, x_center, box:[x,y,w,h]}
    使用 CLASS_CONF_THRES 为不同类别应用独立的置信度阈值。
    """
    outputs = _standardize_outputs(output)
    rows, channels = outputs.shape
    num_classes = channels - 4

    detections = []
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for i in range(rows):
        row = outputs[i]
        cls_scores = row[4:4 + num_classes]

        # 先确定类别，再按类别阈值判断是否保留
        class_id = int(np.argmax(cls_scores))
        # classes 可能是字典（id->name），按原始代码用下标访问
        class_name = classes[class_id] if isinstance(classes, (list, tuple)) else classes.get(class_id, str(class_id))
        max_conf = float(cls_scores[class_id])

        # 按类别阈值过滤
        if max_conf < get_conf_thres_for_class(class_name):
            continue

        x, y, w, h = row[0], row[1], row[2], row[3]
        left = int((x - w / 2) * x_factor)
        top = int((y - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        
        if width <= 2 or height <= 2:
            continue
        if left < 0 or top < 0 or left + width > img_width or top + height > img_height:
            continue

        x_center = left + width / 2
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': max_conf,
            'box': [left, top, width, height],
            'x_center': x_center
        })

    # NMS 去重：不再用全局阈值二次过滤，confidence_threshold 设为 0.0
    if detections:
        boxes = [d['box'] for d in detections]
        scores = [d['confidence'] for d in detections]
        indices = custom_NMSBoxes(boxes, scores, 0.0, IOU_THRES)
        detections = [detections[i] for i in indices]

    return detections

# ========== 核心逻辑：重新设计的分组策略 ==========
def group_detections_by_bin(detections):
    """
    以垃圾桶类别为基准建组，将状态与垃圾分配到最近的垃圾桶（阈值内）
    """
    states = []
    bin_types = []
    wastes = []
    
    for det in detections:
        cls = det['class_name']
        if cls in STATE_CLASSES:
            states.append(det)
        elif cls in BIN_TYPE_CLASSES:
            bin_types.append(det)
        elif cls in WASTE_TO_BIN_TYPE:
            wastes.append(det)
    
    if not bin_types:
        return []
    
    bins = []
    for bin_det in bin_types:
        bin_center = bin_det['x_center']
        nearby_states = [s for s in states if abs(s['x_center'] - bin_center) < X_CENTER_THRESHOLD]
        nearby_wastes = [w for w in wastes if abs(w['x_center'] - bin_center) < X_CENTER_THRESHOLD]
        bins.append({
            'center': bin_center,
            'bin_type': bin_det,
            'states': nearby_states,
            'wastes': nearby_wastes
        })
    
    return bins

def analyze_bin(bin_group):
    """
    分析单个垃圾桶的状态并判断投放正确性（每类取置信度最高的一个）
    """
    bin_type_raw = bin_group['bin_type']['class_name']
    bin_type = BIN_TYPE_CLASSES.get(bin_type_raw, "未知")
    
    if bin_group['states']:
        best_state = max(bin_group['states'], key=lambda x: x['confidence'])
        state = best_state['class_name']
    else:
        state = 'close'
    
    waste_names = []
    if bin_group['wastes']:
        best_waste = max(bin_group['wastes'], key=lambda x: x['confidence'])
        waste_names = [best_waste['class_name']]
    
    waste_names_cn = [WASTE_NAME_CN.get(w, w) for w in waste_names]
    
    is_correct = True
    if waste_names:
        for waste in waste_names:
            correct_bin = WASTE_TO_BIN_TYPE.get(waste, "未知")
            if correct_bin != bin_type:
                is_correct = False
                break
    
    state_cn = "打开" if state == "open" else "关闭"
    waste_str = "、".join(waste_names_cn) if waste_names_cn else "无"
    correct_str = "正确" if is_correct else "错误"
    
    message = f"{bin_type}垃圾桶状态为{state_cn}，垃圾桶里的垃圾为{waste_str}，投放{correct_str}。"
    
    return {
        'state': state,
        'bin_type': bin_type,
        'wastes': waste_names_cn,
        'is_correct': is_correct,
        'message': message
    }

def draw_detections_on_image(img, detections, color_palette):
    """
    在图片上绘制检测框
    """
    for det in detections:
        x, y, w, h = det['box']
        color = color_palette[det['class_id']]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - lh - 10), (x + lw, y), color, cv2.FILLED)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img

# ========== 主函数 ==========
def main():
    # 1. 加载模型和类别
    with open(DATASET_YAML_PATH, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    classes = dataset_config['names']
    
    # 初始化垃圾类别集合
    global WASTE_CLASSES
    WASTE_CLASSES = set(classes.values()) - STATE_CLASSES - set(BIN_TYPE_CLASSES.keys())
    
    print(f"加载 {len(classes)} 个类别")
    print(f"  开合状态: {STATE_CLASSES}")
    print(f"  垃圾桶类别: {set(BIN_TYPE_CLASSES.keys())}")
    print(f"  垃圾种类: {len(WASTE_CLASSES)} 种")
    print(f"  X_CENTER_THRESHOLD: {X_CENTER_THRESHOLD} 像素\n")
    
    # 2. 初始化 ONNX 模型
    session = ort.InferenceSession(MODEL_PATH, providers=get_providers())
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width, input_height = input_shape[2], input_shape[3]
    
    # 3. 收集输入图片
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']:
        for p in in_dir.glob(ext):
            files.add(p.resolve())
    images = sorted(list(files))
    
    print(f"找到 {len(images)} 张图片")
    print(f"输出目录: {out_dir.resolve()}\n")
    
    # 生成颜色调色板
    color_palette = np.random.uniform(100, 255, size=(len(classes), 3))
    
    # 4. 批量推理
    results_log = []
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] 推理: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  跳过：无法读取\n")
            continue
        
        # 推理
        img_data, img_height, img_width = preprocess(img, input_width, input_height)
        outputs = session.run(None, {model_inputs[0].name: img_data})
        detections = postprocess_detections(outputs, input_width, input_height, img_width, img_height, classes)
        
        print(f"  检测到 {len(detections)} 个目标")
        
        # 统计各类别检测数量（调试信息）
        bin_type_detections = [d for d in detections if d['class_name'] in BIN_TYPE_CLASSES]
        state_detections = [d for d in detections if d['class_name'] in STATE_CLASSES]
        waste_detections = [d for d in detections if d['class_name'] in WASTE_TO_BIN_TYPE]
        
        print(f"  垃圾桶类别: {len(bin_type_detections)} 个")
        for d in bin_type_detections:
            print(f"    - {d['class_name']} (conf={d['confidence']:.2f}, x={d['x_center']:.0f})")
        
        print(f"  开合状态: {len(state_detections)} 个")
        print(f"  垃圾种类: {len(waste_detections)} 个")
        
        # 分组分析（以垃圾桶为基准）
        bins = group_detections_by_bin(detections)
        
        print(f"  创建 {len(bins)} 个垃圾桶组")
        for idx, bin_group in enumerate(bins, 1):
            print(f"    组 {idx}: {bin_group['bin_type']['class_name']} "
                  f"(center={bin_group['center']:.0f}, "
                  f"states={len(bin_group['states'])}, "
                  f"wastes={len(bin_group['wastes'])})")
        
        # 分析每个垃圾桶
        valid_bins = []
        for bin_group in bins:
            result = analyze_bin(bin_group)
            valid_bins.append(result)
        
        print(f"  输出 {len(valid_bins)} 个垃圾桶结果\n")
        
        # 输出每个识别到的垃圾桶
        for j, result in enumerate(valid_bins, 1):
            print(f"图片: {img_path.name}")
            print(f"垃圾桶 {j}: {result['message']}\n")
            results_log.append({
                'image': img_path.name,
                'bin_index': j,
                'result': result
            })
        
        # 绘制检测框并保存
        output_img = draw_detections_on_image(img.copy(), detections, color_palette)
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), output_img)
        
        print()
    
    # 5. 保存结果日志（写到 piper 监控目录）
    ensure_dirs()
    log_path = VOICE_RUNTIME_DIR / "detection_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        if not results_log:
            f.write("未检测到垃圾桶或相关目标。\n")
        else:
            for entry in results_log:
                f.write(f"{entry['result']['message']}\n")

    print(f"检测完成！结果已保存到 {log_path}")

if __name__ == '__main__':
    main()
