#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).parent.resolve()

class PlateNumberSpeaker:
    def __init__(self, model="./zh_CN-huayan-medium.onnx", piper_dir="./"):
        # 解析为绝对路径，避免 cwd 影响
        self.model = str((Path(model) if Path(model).is_absolute() else (BASE_DIR / model)).resolve())
        self.piper_dir = str((Path(piper_dir) if Path(piper_dir).is_absolute() else (BASE_DIR / piper_dir)).resolve())

        # 优先使用 piper_dir 内的 piper；找不到则回退 PATH 中的 piper
        piper_in_dir = os.path.join(self.piper_dir, "piper")
        self.piper_cmd = piper_in_dir if os.path.exists(piper_in_dir) else shutil.which("piper") or piper_in_dir

        self.letter_map = {
            'A': 'A', 'B': '必', 'C': '西', 'D': '帝', 'E': '意',
            'F': '艾弗', 'G': '吉', 'H': 'A曲', 'I': '爱', 'J': '杰',
            'K': '开', 'L': '艾勒', 'M': '艾姆', 'N': '恩', 'O': '欧',
            'P': '批', 'Q': '克油', 'R': '啊儿', 'S': '艾斯', 'T': '替',
            'U': '优', 'V': '维', 'W': '达不溜', 'X': '艾克斯', 'Y': '歪',
            'Z': '贼'
        }
        self.number_map = {
            '0': '零', '1': '幺', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        # 汉字读音映射（按需扩展）
        self.char_map = {
            '为': '围',
            '倒': '捣',
        }

    def convert_plate_number(self, text: str) -> str:
        # 统一全角/半角，增强匹配稳定性
        text = unicodedata.normalize("NFKC", text)
        result = []
        for line in text.strip().split('\n'):
            # 先应用汉字映射
            for original, replacement in self.char_map.items():
                line = line.replace(original, replacement)

            # 连续数字拆分读音，单个数字也映射
            def split_numbers(match):
                num_str = match.group(0)
                return ' '.join([self.number_map.get(d, d) for d in num_str])
            line = re.sub(r'\d{2,}', split_numbers, line)
            line = re.sub(r'\d', lambda m: self.number_map.get(m.group(0), m.group(0)), line)

            # 字母：支持大小写（统一转大写查表）
            def convert_letter(match):
                letter = match.group(0).upper()
                return self.letter_map.get(letter, letter)
            line = re.sub(r'[A-Za-z]', convert_letter, line)

            # 车牌分词：省份 + 首字母拼音 + 数字读音（保留原逻辑）
            plate_pattern = r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领学])([诶比西迪衣艾弗吉艾曲爱杰开艾勒艾姆恩欧皮丘艾儿艾斯提优维达不溜艾克斯歪贼德])([零幺二三四五六七八九\s]+)'
            def add_spaces(match):
                province = match.group(1)
                first_letter = match.group(2)
                rest = match.group(3).replace(' ', '')
                rest_spaced = ' '.join(rest)
                return f"{province} {first_letter} {rest_spaced}"
            line = re.sub(plate_pattern, add_spaces, line)

            result.append(line)
        return '\n'.join(result)

    def split_sentences(self, text: str) -> List[str]:
        # 统一分隔符，确保 Piper 正确识别句子边界
        text = text.replace('，', '。').replace(',', '。')
        text = text.replace('；', '。').replace(';', '。')
        text = text.replace('：', '。').replace(':', '。')
        parts = re.split(r'[。！？\.\!\?]+', text)
        return [s.strip() for s in parts if s.strip()]

    # 新增：把"车牌号为：XXX"分成"前缀一句 + 逐字符若干句"
    def segment_plate(self, raw_line: str) -> Tuple[List[str], List[str]]:
        """
        返回 (prefix_sentences, token_sentences)
        - prefix_sentences：车牌前缀（如"1号停车场车牌号为："）
        - token_sentences：车牌内每个字符/字母/数字单独成句（按原字符顺序）
        未匹配车牌格式时，返回([], [])
        """
        # 改进正则：支持"车牌号"后跟冒号（中英文都支持），然后是汉字/字母/数字的任意混合
        # 例如"车牌号为:蒙NZA772" 或 "车牌号：京A12345" 或 "车牌号为： 蒙NZA772"
        # 模式说明：
        # (.*?(?:车牌|车牌号).*?[：:])  - 前缀（包含"车牌"/"车牌号"和冒号）
        # \s*                           - 冒号后可能有空格
        # ([A-Z\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5]*)  - 车牌号（首字大写或汉字，后续任意混合）
        # \s*$                          - 结尾可能有空格
        m = re.search(
            r'(.*?(?:车牌|车牌号).*?[：:])\s*([A-Z\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5]*)\s*$',
            raw_line,
            re.IGNORECASE
        )
        
        if not m:
            print(f"[DEBUG segment_plate] 正则未匹配: {repr(raw_line)}", file=sys.stderr)
            return [], []
        
        prefix = m.group(1)  # 含冒号
        plate_str = m.group(2)
        
        print(f"[DEBUG segment_plate] 匹配成功! prefix={repr(prefix)}, plate_str={repr(plate_str)}", file=sys.stderr)

        # 前缀走原有转换逻辑（数字读幺等）
        prefix_text = self.convert_plate_number(prefix)

        # 逐字符转读音（停顿按原字符分组）
        tokens = []
        for ch in plate_str:
            if ch.isdigit():
                tokens.append(self.number_map.get(ch, ch))
            elif ch.isalpha():
                tokens.append(self.letter_map.get(ch.upper(), ch))
            else:
                # 汉字（如 蒙、学），保留原字作为一个组
                tokens.append(ch)

        print(f"[DEBUG segment_plate] 逐字符映射结果: {tokens}", file=sys.stderr)
        return [prefix_text], tokens

    # 复用：一次性合成并播放，使用 --sentence_silence 控制间隔
    def _speak_batch(self, sentences: List[str], sentence_silence: float) -> bool:
        # 末尾标点与清理
        norm = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not s.endswith(('。', '！', '？', '.', '!', '?')):
                s += '。'
            norm.append(s)
        if not norm:
            return True
        
        print(f"[DEBUG _speak_batch] 最终送给 piper 的文本: {norm}", file=sys.stderr)
        
        if not self.piper_cmd or (not os.path.isabs(self.piper_cmd) and shutil.which(self.piper_cmd) is None) or (os.path.isabs(self.piper_cmd) and not os.path.exists(self.piper_cmd)):
            print(f"错误: 未找到 piper 可执行文件：{self.piper_cmd}", file=sys.stderr)
            return False

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        try:
            proc = subprocess.Popen(
                [self.piper_cmd, "--model", self.model, "--output_file", tmp_wav_path,
                 "--sentence_silence", str(sentence_silence)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.piper_dir
            )
            input_text = "\n".join(norm)
            stdout, stderr = proc.communicate(input=input_text, timeout=120)
            if proc.returncode != 0:
                print(f"[ERROR] Piper 失败: {stderr.strip() if stderr else '未知错误'}", file=sys.stderr)
                try: os.remove(tmp_wav_path)
                except Exception: pass
                return False

            # 播放（Linux 常见为 aplay，或 ffplay）
            if shutil.which("aplay"):
                rc = subprocess.run(["aplay", "-q", tmp_wav_path]).returncode
            elif shutil.which("ffplay"):
                rc = subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", tmp_wav_path]).returncode
            else:
                print(f"提示: 未找到 aplay/ffplay，已生成音频：{tmp_wav_path}", file=sys.stderr)
                rc = 0
            return rc == 0
        except subprocess.TimeoutExpired:
            print("[ERROR] 合成/播放超时", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return False
        finally:
            try: os.remove(tmp_wav_path)
            except Exception: pass

    def speak_text(self, text: str, background: bool = False,
                   sentence_silence: float = 0.1, token_silence: float = None) -> bool:
        """
        - sentence_silence：普通句子之间的静音
        - token_silence：车牌逐字符之间的静音（不传则用 sentence_silence）
        """
        token_silence = sentence_silence if token_silence is None else token_silence

        ok_all = True
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            print(f"[DEBUG speak_text] 处理行: {repr(raw_line)}", file=sys.stderr)

            # 尝试按"车牌"格式分段
            prefix_sents, token_sents = self.segment_plate(raw_line)
            
            if prefix_sents and token_sents:
                print(f"[DEBUG speak_text] 走车牌分段逻辑: prefix={prefix_sents}, tokens={token_sents}", file=sys.stderr)
                ok_all &= self._speak_batch(prefix_sents, sentence_silence)
                ok_all &= self._speak_batch(token_sents, token_silence)
                continue

            # 否则按整体转换并播报
            print(f"[DEBUG speak_text] 走整体转换逻辑", file=sys.stderr)
            processed = self.convert_plate_number(raw_line)
            print(f"[DEBUG speak_text] 转换后: {repr(processed)}", file=sys.stderr)
            normal_sents = self.split_sentences(processed)
            print(f"[DEBUG speak_text] split_sentences={normal_sents}", file=sys.stderr)
            ok_all &= self._speak_batch(normal_sents, sentence_silence)

        return ok_all

    def speak_file(self, file_path: str, sentence_silence: float = 0.1, token_silence: float = None) -> bool:
        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在 - {file_path}", file=sys.stderr)
            return False
        
        # 统一编码读取，处理可能的 BOM
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            text = f.read()
        
        print(f"[DEBUG speak_file] 读取文件内容: {repr(text)}", file=sys.stderr)
        return self.speak_text(text, sentence_silence=sentence_silence, token_silence=token_silence)

if __name__ == "__main__":
    # 参数通过环境变量指定
    DEFAULT_MODEL = str((BASE_DIR / "zh_CN-huayan-medium.onnx").resolve())
    DEFAULT_PIPER_DIR = str(BASE_DIR)
    DEFAULT_PAUSE = 0.1         # 普通句子间隔
    DEFAULT_TOKEN_PAUSE = 0.07  # 车牌内逐字符间隔（按原字符分组）

    model = os.getenv("SPEAK_MODEL", DEFAULT_MODEL)
    piper_dir = os.getenv("PIPER_DIR", DEFAULT_PIPER_DIR)
    # 允许通过 SPEAK_PAUSE 覆盖
    try:
        pause_time = float(os.getenv("SPEAK_PAUSE", str(DEFAULT_PAUSE)))
    except ValueError:
        pause_time = DEFAULT_PAUSE
    try:
        token_pause = float(os.getenv("TOKEN_PAUSE", str(DEFAULT_TOKEN_PAUSE)))
    except ValueError:
        token_pause = DEFAULT_TOKEN_PAUSE

    if len(sys.argv) < 2:
        print("[ERROR] 需要提供输入文件路径或文本内容", file=sys.stderr)
        sys.exit(1)

    input_arg = sys.argv[1]
    speaker = PlateNumberSpeaker(model=model, piper_dir=piper_dir)

    # 关键环境打印，便于核对是否用的同一套资源
    print(f"[speak_cp] cwd={os.getcwd()} piper={speaker.piper_cmd} piper_dir={speaker.piper_dir} model={speaker.model} sentence_pause={pause_time} token_pause={token_pause}", file=sys.stderr)

    if os.path.isfile(input_arg):
        ok = speaker.speak_file(input_arg, sentence_silence=pause_time, token_silence=token_pause)
    else:
        ok = speaker.speak_text(input_arg, sentence_silence=pause_time, token_silence=token_pause)

    sys.exit(0 if ok else 1)
