#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional

# ===================== 运行时常量（不再命令行传参） =====================
MONITOR_DIR = "./monitor_txt"    # 修改为要监控的目录
CHECK_INTERVAL = 0.5             # 轮询间隔（秒）

# ===================== 可选覆盖配置（按文件名覆盖默认配置） =====================
FILE_SPEAK_CONFIG = {
    # 对特定编号使用车牌读法脚本
    "5.txt": {"speak_script": "./speak_cp.py"},
    "6.txt": {"speak_script": "./speak_cp.py"},
    "8.txt": {"speak_script": "./speak_cp.py"},
}

DEFAULT_CONFIG = {
    "speak_script": "./speak.py",
    "piper_model": "./zh_CN-huayan-medium.onnx",
    "pause_time": 0.1,     # 句间停顿时间（秒）
    "token_pause": 0.07,   # 车牌内逐字符间隔（秒）
    "piper_dir": "./"      # piper 可执行所在目录；若在 PATH 中，可保留 "./"
}

# 基于脚本目录解析相对路径，避免受当前工作目录影响
BASE_DIR = Path(__file__).parent.resolve()
def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (BASE_DIR / pp).resolve()

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(resolve_path('piper_monitor.log')), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class PiperFileMonitor:
    def __init__(self, monitor_dir: str, check_interval: float = 0.5):
        # 监控目录也用脚本目录为基准
        self.monitor_dir = resolve_path(monitor_dir)
        self.check_interval = check_interval
        # 覆盖键统一小写，避免大小写差异
        self._override_map = {str(k).lower(): v for k, v in FILE_SPEAK_CONFIG.items()}
        self._validate_config()

    def _validate_config(self):
        if not self.monitor_dir.exists():
            logger.error(f"监控目录不存在: {self.monitor_dir}")
            sys.exit(1)

        # 校验默认脚本/模型存在（相对脚本目录解析）
        default_script = resolve_path(DEFAULT_CONFIG["speak_script"])
        if not default_script.exists():
            logger.error(f"默认播报脚本不存在: {default_script}")
            sys.exit(1)
        default_model = resolve_path(DEFAULT_CONFIG["piper_model"])
        if not default_model.exists():
            logger.warning(f"默认模型文件不存在: {default_model}（播报时可能出错）")

        # 校验可选覆盖配置
        for filename, cfg in FILE_SPEAK_CONFIG.items():
            merged = {**DEFAULT_CONFIG, **cfg}
            script_path = resolve_path(merged["speak_script"])
            if not script_path.exists():
                logger.error(f"覆盖配置脚本不存在: [{filename}] -> {script_path}")
                sys.exit(1)
            model_path = resolve_path(merged["piper_model"])
            if not model_path.exists():
                logger.warning(f"覆盖配置模型不存在: [{filename}] -> {model_path}")

    def _get_file_config(self, filename: str) -> dict:
        key = filename.lower()
        file_cfg = self._override_map.get(key, {})
        logger.info(f"是否命中文件覆盖({filename}): {'是' if file_cfg else '否'}")
        full_cfg = {**DEFAULT_CONFIG, **file_cfg}
        # 统一把路径解析为基于脚本目录的绝对路径
        full_cfg["speak_script"] = str(resolve_path(full_cfg["speak_script"]))
        full_cfg["piper_model"] = str(resolve_path(full_cfg["piper_model"]))
        full_cfg["piper_dir"]   = str(resolve_path(full_cfg["piper_dir"]))
        return full_cfg

    def _find_next_txt(self) -> Optional[Path]:
        # 扫描目录中所有 *.txt，按“纯数字文件名”的整数值升序，取最小
        candidates = []
        for p in self.monitor_dir.glob("*.txt"):
            name = p.stem
            try:
                n = int(name)  # 仅接受 1.txt/2.txt/… 这类纯数字命名
                candidates.append((n, p))
            except ValueError:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _call_speak_script(self, file_path: Path, config: dict) -> bool:
        try:
            logger.info(f"\n开始播报文件: {file_path.name}")
            logger.info(f"使用脚本: {config['speak_script']}")
            logger.info(f"使用模型: {config['piper_model']}")
            logger.info(f"piper_dir: {config['piper_dir']}")
            logger.info(f"句间停顿: {config['pause_time']}秒, token停顿: {config.get('token_pause', config['pause_time'])}秒")

            env = os.environ.copy()
            env["SPEAK_MODEL"] = config["piper_model"]
            env["SPEAK_PAUSE"] = str(config["pause_time"])
            env["PIPER_DIR"]   = config["piper_dir"]
            # 供 speak_cp.py 使用：若未显式设置，则用 pause_time
            env["TOKEN_PAUSE"] = str(config.get("token_pause", config["pause_time"]))

            cmd = [sys.executable, config["speak_script"], str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

            if result.returncode == 0:
                logger.info(f"文件播报完成: {file_path.name}")
                if result.stdout:
                    logger.info(result.stdout.strip())
                return True
            logger.error(f"播报脚本失败 code={result.returncode}")
            if result.stderr:
                logger.error(result.stderr.strip())
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"播报超时: {file_path.name}")
            return False
        except Exception as e:
            logger.error(f"调用播报脚本异常: {e}", exc_info=True)
            return False

    def start_monitor(self):
        logger.info("="*60)
        logger.info("Piper文件监控播报程序启动（最小序号优先）")
        logger.info(f"监控目录: {self.monitor_dir}")
        logger.info(f"检测间隔: {self.check_interval}秒")
        logger.info("="*60)

        while True:
            next_file = self._find_next_txt()
            if not next_file:
                time.sleep(self.check_interval)
                continue

            # 可选：确保写入完成（大小两次一致）
            try:
                size1 = next_file.stat().st_size
                time.sleep(0.2)
                size2 = next_file.stat().st_size
                if size1 != size2:
                    time.sleep(self.check_interval)
                    continue
            except Exception:
                pass

            file_cfg = self._get_file_config(next_file.name)
            logger.info(f"检测到最小序号文件: {next_file.name}")
            success = self._call_speak_script(next_file, file_cfg)

            try:
                if success:
                    processed_dir = self.monitor_dir / "processed"
                    processed_dir.mkdir(exist_ok=True)
                    target = processed_dir / next_file.name

                    if target.exists():
                        # 先把旧文件改名为“时间戳-原名”，再把新文件放原名
                        import datetime, uuid
                        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                        old_renamed = processed_dir / f"{ts}-{uuid.uuid4().hex[:6]}-{next_file.name}"
                        try:
                            target.rename(old_renamed)
                            logger.info(f"已重命名已存在的旧文件为: {old_renamed.name}")
                        except FileNotFoundError:
                            pass

                    # 将“新文件”按原名放入 processed
                    os.replace(next_file, target)
                    logger.info(f"播报完成，已移动到: {target}")
                else:
                    failed_dir = self.monitor_dir / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    import datetime
                    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    target = failed_dir / f"{ts}-{next_file.name}"
                    os.replace(next_file, target)
                    logger.warning(f"播报失败，已移动到: {target}")
            except Exception as e:
                logger.error(f"移动文件失败: {next_file}，错误: {e}")

def main():
    monitor = PiperFileMonitor(
        monitor_dir=MONITOR_DIR,
        check_interval=CHECK_INTERVAL
    )
    try:
        monitor.start_monitor()
    except KeyboardInterrupt:
        logger.info("收到中断信号，已停止监控。")

if __name__ == "__main__":
    main()
