import numpy as np
import onnxruntime as rt
import cv2
import os

def keepratio_resize(img):
    mask_height = 32
    mask_width = 804
    h, w = img.shape[:2]
    cur_ratio = w / float(h)
    if cur_ratio > float(mask_width) / mask_height:
        cur_target_height = mask_height
        cur_target_width = mask_width
    else:
        cur_target_height = mask_height
        cur_target_width = int(mask_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    mask = np.zeros([mask_height, mask_width, 3], dtype=np.uint8)
    mask[:img.shape[0], :img.shape[1], :] = img
    return mask

def ocr_recognize(image_path, model_path='model.onnx', vocab_path='vocab.txt'):
    """
    OCR 识别函数
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
        vocab_path: 字典路径
    
    Returns:
        str: 识别结果（取最长的）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'无法读取图片: {image_path}')
    
    img = keepratio_resize(img)
    
    if img.shape[1] < 300:
        pad = np.zeros((img.shape[0], 300 - img.shape[1], 3), dtype=img.dtype)
        img = np.concatenate([img, pad], axis=1)
    
    # 切片
    chunk_img = []
    for i in range(3):
        left = (300 - 48) * i
        right = left + 300
        crop = img[:, left:right, :]
        if crop.shape[1] < 300:
            pad = np.zeros((crop.shape[0], 300 - crop.shape[1], 3), dtype=crop.dtype)
            crop = np.concatenate([crop, pad], axis=1)
        chunk_img.append(crop)
    
    merge_img = np.concatenate(chunk_img, axis=0)
    data = merge_img.reshape(3, 32, 300, 3) / 255.0
    data = data.transpose(0, 3, 1, 2).astype(np.float32)
    
    # 推理
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    res = sess.run([output_name], {input_name: data})
    
    # CTC 解码
    preds = np.argmax(res[0], axis=-1)
    
    # 加载字典
    labelMapping = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        cnt = 2
        for line in f:
            labelMapping[cnt] = line.strip('\n')
            cnt += 1
    
    # 解码
    final_str_list = []
    batchSize, length = preds.shape
    for i in range(batchSize):
        pred_idx = preds[i].tolist()
        last_p = 0
        str_pred = []
        for p in pred_idx:
            if p != last_p and p != 0:
                str_pred.append(labelMapping.get(p, ''))
            last_p = p
        final_str_list.append(''.join(str_pred))
    
    # 返回最长的识别结果
    return max(final_str_list, key=len) if final_str_list else "未知建筑"

# 原有测试代码
if __name__ == '__main__':
    result = ocr_recognize('C:/Users/26233/Desktop/fire_title/title_crops/f_t_1.jpg')
    print(result)