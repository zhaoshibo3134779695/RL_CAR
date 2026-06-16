#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import time
import shutil
import tempfile

class SentencePiperSpeaker:
    def __init__(self, model="./zh_CN-huayan-medium.onnx", piper_dir="./"):
        self.model = model
        self.piper_dir = piper_dir
        self.piper_cmd = os.path.join(piper_dir, "piper")
        
        # 自定义字符读音映射（按需增减）
        self.char_map = {
            '倒': '捣',
            '为': '围',
            '关': '观',
            '方': '妨',
            '厦': '煞',
            'B': '必',
        }

    def apply_char_map(self, text: str) -> str:
        """应用读音映射"""
        for src, dst in self.char_map.items():
            text = text.replace(src, dst)
        return text

    def split_sentences(self, text: str):
        # 将逗号、分号、冒号都视为“句子分隔”
        text = text.replace('，', '。')
        text = text.replace(',', '。')
        text = text.replace('；', '。')
        text = text.replace(';', '。')
        text = text.replace('：', '。')
        text = text.replace(':', '。')
        # 同时支持中英文句末标点
        parts = re.split(r'[。！？\.\!\?]+', text)
        return [s.strip() for s in parts if s.strip()]

    def speak_batch(self, sentences, sentence_silence=0.05) -> bool:
        """
        一次性合成所有句子，使用 --sentence_silence 控制句间静音；
        可显著减少每句进程启动/播放器缓冲导致的“假停顿”。
        """
        # 预处理与末尾标点
        norm = []
        for s in sentences:
            s = self.apply_char_map(s.strip())
            if not s:
                continue
            if not s.endswith(('。', '！', '？', '.', '!', '?')):
                s += '。'
            norm.append(s)
        if not norm:
            print("无可播报内容")
            return False

        if not os.path.exists(self.piper_cmd):
            print(f"错误: 未找到 piper 可执行文件：{self.piper_cmd}")
            return False

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        try:
            # 一次性传入所有句子
            cmd = [
                self.piper_cmd,
                "--model", self.model,
                "--output_file", tmp_wav_path,
                "--sentence_silence", str(sentence_silence)
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.piper_dir
            )
            input_text = "\n".join(norm)
            stdout, stderr = proc.communicate(input=input_text, timeout=120)
            if proc.returncode != 0:
                if stderr:
                    print(f"Piper 错误: {stderr.strip()}")
                try: os.remove(tmp_wav_path)
                except Exception: pass
                return False

            # 播放（aplay 优先，ffplay 兜底）
            if shutil.which("aplay"):
                rc = subprocess.run(["aplay", tmp_wav_path]).returncode
            elif shutil.which("ffplay"):
                rc = subprocess.run(
                    ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", tmp_wav_path]
                ).returncode
            else:
                print(f"提示: 未找到 aplay/ffplay，已生成音频：{tmp_wav_path}")
                rc = 0

            return rc == 0
        except subprocess.TimeoutExpired:
            print("错误: 合成超时")
            return False
        except Exception as e:
            print(f"错误: {e}")
            return False
        finally:
            try: os.remove(tmp_wav_path)
            except Exception: pass

    # 保留逐句模式（备用）
    def speak_sentences(self, sentences, pause_time=0.1):
        total = len(sentences)
        print(f"共 {total} 个句子待播报")
        print("=" * 50)
        for i, sentence in enumerate(sentences, 1):
            print(f"[{i}/{total}] {sentence}")
            if not self.speak_single_sentence(sentence):
                print("播报失败")
                continue
            if i < total:
                time.sleep(pause_time)
        print("=" * 50)
        print("全部播报完成")

    def speak_single_sentence(self, sentence: str) -> bool:
        sentence = self.apply_char_map(sentence)
        if not sentence.endswith(('。', '！', '？', '.', '!', '?')):
            sentence += '。'
        if not os.path.exists(self.piper_cmd):
            print(f"错误: 未找到 piper 可执行文件：{self.piper_cmd}")
            return False

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        try:
            proc = subprocess.Popen(
                [self.piper_cmd, "--model", self.model, "--output_file", tmp_wav_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.piper_dir
            )
            stdout, stderr = proc.communicate(input=sentence, timeout=20)
            if proc.returncode != 0:
                if stderr:
                    print(f"Piper 错误: {stderr.strip()}")
                try: os.remove(tmp_wav_path)
                except Exception: pass
                return False

            if shutil.which("aplay"):
                rc = subprocess.run(["aplay", tmp_wav_path]).returncode
            elif shutil.which("ffplay"):
                rc = subprocess.run(
                    ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", tmp_wav_path]
                ).returncode
            else:
                print(f"提示: 未找到 aplay/ffplay，已生成音频：{tmp_wav_path}")
                rc = 0
            return rc == 0
        except subprocess.TimeoutExpired:
            print("错误: 当前句子播报超时（超过 20 秒）")
            return False
        except Exception as e:
            print(f"错误: {e}")
            return False
        finally:
            try: os.remove(tmp_wav_path)
            except Exception: pass

    def speak_file(self, file_path, pause_time=0.1):
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return False
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = self.split_sentences(text)
        # 批量模式：pause_time 等价于句间静音秒数
        return self.speak_batch(sentences, sentence_silence=pause_time)

    def speak_text(self, text, pause_time=0.1):
        sentences = self.split_sentences(text)
        return self.speak_batch(sentences, sentence_silence=pause_time)

if __name__ == "__main__":
    # 参数通过环境变量指定
    DEFAULT_MODEL = "./zh_CN-huayan-medium.onnx"
    DEFAULT_PIPER_DIR = "./"
    DEFAULT_PAUSE = 0.2  # 句间静音秒数（用于批量模式）

    model = os.getenv("SPEAK_MODEL", DEFAULT_MODEL)
    piper_dir = os.getenv("PIPER_DIR", DEFAULT_PIPER_DIR)
    pause_time = DEFAULT_PAUSE

    if len(sys.argv) < 2:
        print("错误: 需要提供输入文件路径或文本内容")
        sys.exit(1)

    input_arg = sys.argv[1]
    speaker = SentencePiperSpeaker(model=model, piper_dir=piper_dir)

    if os.path.isfile(input_arg):
        ok = speaker.speak_file(input_arg, pause_time=pause_time)
    else:
        ok = speaker.speak_text(input_arg, pause_time=pause_time)

    sys.exit(0 if ok else 1)
