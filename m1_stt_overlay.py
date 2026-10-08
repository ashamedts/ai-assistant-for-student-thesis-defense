# -*- coding: utf-8 -*-
"""M1 演示：麦克风 → 俄语识别（faster-whisper 本地）→ 离线翻译（NLLB）→ 置顶悬浮窗。

运行方式（二选一）：
  带日志:  <你的venv>/Scripts/python.exe m1_stt_overlay.py
  无窗口:  <你的venv>/Scripts/pythonw.exe m1_stt_overlay.py

快捷键：
  Esc       隐藏 / 显示悬浮窗（投影时应急）
  Ctrl+Q    退出
  窗口本身可按住拖动，双击右上角"×"退出。

说明：本程序只在你本机显示，与 PPT 演讲者视图同屏；投影仪不会出现。
"""
import difflib
import glob
import json
import os
import queue
import re
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt6.QtCore import Qt, QTimer, QMetaObject, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from nllb_translate import OfflineTranslator
import qwen_fix

SAMPLE_RATE = 16000          # whisper 需要的采样率
BLOCK_SIZE = 800             # 50ms 一块
SPEECH_RMS = 300.0           # 说话音量阈值（int16 域），教室吵就调高
SILENCE_TAIL = 1.2           # 静音持续多久算一句话结束（秒），TTS/句中停顿更短不会误切
MIN_SPEECH = 0.5             # 最短说话时长（秒），过滤咳嗽等
MIN_SEGMENT = 2.0            # 太短的片段直接丢弃（短碎片最容易让 whisper 幻觉）
MAX_SEGMENT = 12.0           # 最长说话时长（秒），长问题也不会被强切（老师提问常在 8-12 秒）

# whisper 在无语音/残缺音频上的常见幻觉模板（命中即丢弃）
HALLUC_PATTERNS = (
    "продолжение следует", "субтитры", "dima", "torzok",
    "не волнуйся", "спасибо за просмотр", "подписывайтесь",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_Q = queue.Queue()       # (识别原文, 修正俄语, 中文) 结果队列

LOG_FILE = os.path.join(BASE_DIR, "app.log")
QA_HISTORY_FILE = os.path.join(BASE_DIR, "qa_history.md")
KEEP_RECORDINGS = 60  # 最多保留最近 60 段录音，避免 D 盘被堆满

KEYWORDS = ""
CONTEXT = ""
MATERIALS = []


def load_kb():
    """读取关键词（whisper 引导）与背景（Qwen 纠错上下文），失败留空。"""
    global KEYWORDS, CONTEXT
    try:
        with open(os.path.join(BASE_DIR, "keywords.txt"), encoding="utf-8") as f:
            KEYWORDS = f.read().strip()
    except Exception:
        KEYWORDS = ""
    try:
        with open(os.path.join(BASE_DIR, "context.txt"), encoding="utf-8") as f:
            CONTEXT = f.read().strip()
    except Exception:
        CONTEXT = ""


def load_materials():
    """读取导入的讲稿材料（materials_data.json）。"""
    global MATERIALS
    try:
        with open(os.path.join(BASE_DIR, "materials_data.json"), encoding="utf-8") as f:
            MATERIALS = json.load(f)
        if MATERIALS:
            log(f"已加载讲稿材料 {len(MATERIALS)} 块")
    except Exception:
        MATERIALS = []


def retrieve(query, blocks, k=5):
    """按俄语关键词共现打分，取与提问最相关的 k 块材料。"""
    qw = set(re.findall(r"[а-яё]{3,}", query.lower()))
    if not qw:
        return blocks[:k]
    scored = []
    for b in blocks:
        bw = set(re.findall(r"[а-яё]{3,}", b["text"].lower()))
        score = len(qw & bw) / len(qw)
        scored.append((score, b))
    scored.sort(key=lambda x: -x[0])
    top = [b for s, b in scored[:k] if s > 0]
    return top if top else blocks[:k]


def log(msg):
    """把运行日志追加到 <脚本目录>/app.log（pythonw 无控制台也能留痕）。"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def append_qa(ru, ans_ru, ans_zh):
    """把一次提问/回答追加到 qa_history.md（D 盘），便于课后复盘。"""
    try:
        os.makedirs(os.path.dirname(QA_HISTORY_FILE), exist_ok=True)
        with open(QA_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"提问（识别）: {ru}\n"
                    f"俄语回答: {ans_ru}\n"
                    f"中文理解: {ans_zh}\n")
    except Exception as e:
        log(f"写入问答历史失败: {e}")


def whisper_device():
    """优先 GPU，失败则降级 CPU。"""
    try:
        from ctranslate2 import get_cuda_device_count
        if get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def pick_audio_source(loopback):
    """mic: 返回默认输入设备；loopback: 返回默认输出设备（内录系统播放的声音）。"""
    try:
        devices = sd.query_devices()
        if loopback:
            out = sd.default.device[1]
            if out is not None and devices[out]["max_output_channels"] > 0:
                log(f"内录模式: 使用输出设备 [{out}] {devices[out]['name']}")
                return out
            log("内录模式: 默认输出设备不可用，返回 None")
            return None
        default_in = sd.default.device[0]
        print("可用输入设备：")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                mark = " <- 默认" if i == default_in else ""
                print(f"  [{i}] {d['name']}{mark}")
        if default_in is not None and devices[default_in]["max_input_channels"] > 0:
            log(f"使用默认输入设备: [{default_in}] {devices[default_in]['name']}")
            return default_in
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                log(f"默认输入设备不可用，改用 [{i}] {d['name']}")
                return i
    except Exception as e:
        print("查询设备失败：", e)
        log(f"查询设备失败: {e}")
    return None


class AudioCapture(threading.Thread):
    """采集声音（麦克风或系统内录），按静音切句，把每句话送入识别队列。"""

    def __init__(self, transcribe_q, loopback=False, save_dir=None):
        super().__init__(daemon=True)
        self._tq = transcribe_q
        self._loopback = loopback
        self._save_dir = save_dir
        self._stop = threading.Event()
        self._blocks = queue.Queue()
        self._paused = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # 忽略状态提示（如过载）
        self._blocks.put(indata.copy())

    def _save(self, speech, tag):
        """把切好的句子存成 wav，便于分析识别质量。"""
        if not self._save_dir:
            return
        try:
            os.makedirs(self._save_dir, exist_ok=True)
            name = time.strftime("%m%d_%H%M%S_") + tag + ".wav"
            path = os.path.join(self._save_dir, name)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(speech.tobytes())
            log(f"已保存录音: {path} ({len(speech)/SAMPLE_RATE:.1f}s)")
            self._cleanup_old()
        except Exception as e:
            log(f"保存录音失败: {e}")

    def _cleanup_old(self):
        """删除最早的录音，只保留最近 KEEP_RECORDINGS 段。"""
        try:
            files = sorted(glob.glob(os.path.join(self._save_dir, "*_mic.wav"))
                           + glob.glob(os.path.join(self._save_dir, "*_loop.wav")))
            while len(files) > KEEP_RECORDINGS:
                os.remove(files.pop(0))
        except Exception as e:
            log(f"清理旧录音失败: {e}")

    def stop(self):
        self._stop.set()

    def set_paused(self, paused):
        """暂停/恢复监听：回答问题时按 F9，避免把自己的回答识别进去。"""
        self._paused = paused
        log("监听已暂停" if paused else "监听已恢复")

    def run(self):
        try:
            extra = sd.WasapiSettings(loopback=True) if self._loopback else None
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=BLOCK_SIZE,
                                device=pick_audio_source(self._loopback),
                                extra_settings=extra,
                                callback=self._callback):
                log("采集线程已启动，开始监听声音")
                speech = np.zeros(0, dtype=np.int16)
                silence_s = 0.0
                speech_s = 0.0
                while not self._stop.is_set():
                    try:
                        block = self._blocks.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if self._paused:
                        # 暂停期间不累积语音，避免把自己的回答识别进去
                        speech = np.zeros(0, dtype=np.int16)
                        speech_s = 0.0
                        silence_s = 0.0
                        continue
                    rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                    talking = rms > SPEECH_RMS
                    if talking:
                        speech = np.concatenate([speech, block[:, 0]])
                        speech_s += len(block) / SAMPLE_RATE
                        silence_s = 0.0
                    else:
                        silence_s += len(block) / SAMPLE_RATE
                    if not talking and speech_s >= MIN_SPEECH and silence_s >= SILENCE_TAIL:
                        if speech_s >= MIN_SEGMENT:
                            log(f"切句送入识别: {speech_s:.1f}s")
                            self._save(speech, "loop" if self._loopback else "mic")
                            TEXT_Q.put(("*busy*", "", "", ""))
                            self._tq.put(speech.astype(np.float32) / 32768.0)
                        else:
                            log(f"碎片过短({speech_s:.1f}s)丢弃")
                        speech = np.zeros(0, dtype=np.int16)
                        speech_s = 0.0
                        silence_s = 0.0
                    elif len(speech) >= SAMPLE_RATE * MAX_SEGMENT:  # 连续说话超长，强制切句
                        log(f"超长强制切句: {len(speech)/SAMPLE_RATE:.0f}s")
                        self._save(speech, "loop" if self._loopback else "mic")
                        TEXT_Q.put(("*busy*", "", "", ""))
                        self._tq.put(speech.astype(np.float32) / 32768.0)
                        speech = np.zeros(0, dtype=np.int16)
                        speech_s = 0.0
        except Exception as e:
            log(f"采集线程失败: {type(e).__name__}: {e}")
            print(f"采集线程失败: {type(e).__name__}: {e}")


class TranscribeWorker(threading.Thread):
    """从识别队列取音频，whisper 识别俄语 → NLLB 翻译成中文 → 结果队列。"""

    def __init__(self, transcribe_q, whisper, translator):
        super().__init__(daemon=True)
        self._q = transcribe_q
        self._w = whisper
        self._tr = translator
        self._last_out = ""   # 最近一次输出的回答文本，用于回声过滤

    def run(self):
        while True:
            audio = self._q.get()
            if audio is None:
                break
            try:
                t0 = time.time()
                segments, _ = self._w.transcribe(
                    audio, language="ru", beam_size=6,
                    vad_filter=True, without_timestamps=True,
                    condition_on_previous_text=False,
                    initial_prompt=KEYWORDS or None)
                ru_parts = []
                no_speech = 0.0
                n_seg = 0
                for s in segments:
                    ru_parts.append(s.text.strip())
                    no_speech += s.no_speech_prob
                    n_seg += 1
                ru = " ".join(p for p in ru_parts if p).strip()
                # 幻觉检测：整段被判"无语音"，或命中已知幻觉模板
                avg_no_speech = no_speech / n_seg if n_seg else 1.0
                if (not ru or avg_no_speech > 0.55
                        or any(p in ru.lower() for p in HALLUC_PATTERNS)):
                    log(f"疑似幻觉已拦截 (no_speech={avg_no_speech:.2f}): {ru or '(空)'}")
                    TEXT_Q.put(("*idle*", "", "", ""))
                    continue
                # 回声过滤：和刚才生成的回答几乎相同 → 是自己在读回答，忽略
                if self._last_out:
                    ratio = difflib.SequenceMatcher(
                        None, ru.lower(), self._last_out.lower()).ratio()
                    if ratio > 0.7:
                        log(f"疑似回声已忽略 (相似度{ratio:.2f}): {ru}")
                        TEXT_Q.put(("*idle*", "", "", ""))
                        continue
                # 在线并行识别：仅在离线结果偏短（可能不完整）时等待在线结果，
                # 长句直接跳过——教室网差时不白等，识别结果完整时不需要在线。
                if len(ru.split()) < 6:
                    online_ru = self._online_transcribe(audio, timeout=3)
                    if online_ru and len(online_ru.split()) > len(ru.split()) + 2:
                        log(f"在线识别更完整，采用在线结果: {online_ru}（离线: {ru}）")
                        ru = online_ru
                # 先翻译问题（NLLB，1-2 秒）并立即显示，用户不用干等回答生成
                try:
                    zh_q = self._tr.translate(ru) or ""
                except Exception:
                    zh_q = ""
                TEXT_Q.put(("*question*", ru, "", zh_q))
                # 结合材料生成回答；材料为空或生成失败则降级为纠错+翻译
                if MATERIALS:
                    related = retrieve(ru, MATERIALS, k=3)
                    ans_ru, ans_zh = qwen_fix.answer(ru, CONTEXT, related)
                else:
                    ans_ru, ans_zh = None, None
                if ans_ru is None or ans_zh is None:
                    fixed, zh = qwen_fix.fix(ru, CONTEXT)
                    if fixed is None or zh is None:
                        zh = self._tr.translate(ru)
                        fixed = None
                    ans_ru, ans_zh = fixed or ru, zh
                    log("材料回答不可用，降级为纠错+翻译")
                dt = time.time() - t0
                print(f"[{dt:.1f}s] RU: {ru}\n        ANS: {ans_ru}\n        ZH: {ans_zh}")
                log(f"识别完成 [{dt:.1f}s] RU: {ru} | ANS: {ans_ru} | ZH: {ans_zh}")
                append_qa(ru, ans_ru, ans_zh)
                self._last_out = ans_ru
                TEXT_Q.put((ru, ans_ru, ans_zh, zh_q))
            except Exception as e:
                print("识别/翻译失败：", e)
                log(f"识别/翻译失败: {type(e).__name__}: {e}")
                TEXT_Q.put(("*idle*", "", "", ""))

    def _online_transcribe(self, audio, timeout=3):
        """Google Web Speech API 免费在线识别俄语。无网/失败/超时返回 None。

        作为离线 whisper 的补充：有网时在线识别通常更完整（尤其长句），
        离线结果始终保底，二者择优在主流程完成。
        """
        try:
            import speech_recognition as sr
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            ad = sr.AudioData(pcm, SAMPLE_RATE, 2)
            rec = sr.Recognizer()
            box = {}

            def worker():
                try:
                    box["res"] = rec.recognize_google(ad, language="ru-RU")
                except Exception as e:
                    log(f"在线识别失败: {type(e).__name__}")
                    box["res"] = None

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if "res" in box and box["res"]:
                return box["res"].strip()
            return None
        except Exception as e:
            log(f"在线识别不可用: {type(e).__name__}")
            return None


class Overlay(QWidget):
    """无边框、置顶的悬浮窗（纯色不透明背景，内容全部展开显示）。"""

    def __init__(self):
        super().__init__()
        self._drag_pos = None
        self._history = []
        self._cap = None
        self.setWindowTitle("演讲问答助手")
        # 无边框 + 置顶；不加 Tool 标志，让窗口进任务栏（最小化后可点任务栏图标恢复）
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(660)
        self.setMinimumHeight(150)

        self._raw = QLabel("（识别原文将显示在这里）")
        self._raw.setWordWrap(True)
        self._raw.setStyleSheet("color:#cbd5e1;font-size:13px;")
        self._ru = QLabel("等待提问…")
        self._ru.setWordWrap(True)
        self._ru.setStyleSheet("color:#ffffff;font-size:18px;font-weight:600;")
        self._zh = QLabel("（识别后将生成中文理解）")
        self._zh.setWordWrap(True)
        self._zh.setStyleSheet("color:#fde68a;font-size:16px;")
        self._status = QLabel("● 运行中")
        self._status.setStyleSheet("color:#4ade80;font-size:13px;font-weight:600;")

        quit_btn = QPushButton("×")
        quit_btn.setFixedSize(26, 26)
        quit_btn.setStyleSheet("color:#fff;background:transparent;border:none;font-size:16px;")
        quit_btn.clicked.connect(QApplication.instance().quit)
        min_btn = QPushButton("—")
        min_btn.setFixedSize(26, 26)
        min_btn.setStyleSheet("color:#fff;background:transparent;border:none;font-size:16px;")
        min_btn.clicked.connect(self.showMinimized)
        min_btn.setToolTip("最小化到任务栏（点任务栏图标恢复）")

        head = QHBoxLayout()
        head.addWidget(self._status)
        head.addStretch(1)
        head.addWidget(min_btn)
        head.addWidget(quit_btn)

        # 主内容区：识别原文 / 俄语回答 / 中文理解。
        # 高度固定（280px），内容超出时用滚轮/滚动条查看全部，窗口不会顶出屏幕。
        main_wrap = QWidget()
        mc = QVBoxLayout(main_wrap)
        mc.setContentsMargins(0, 0, 0, 0)
        mc.setSpacing(6)
        mc.addWidget(self._raw)
        mc.addWidget(self._ru)
        mc.addWidget(self._zh)
        mc.addStretch(1)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(main_wrap)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:#1e293b;border-radius:4px;}"
            "QScrollBar::handle:vertical{background:#64748b;border-radius:4px;min-height:24px;}"
            "QScrollBar::add-line:vertical{height:0;}QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}")
        self._scroll.setFixedHeight(280)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sep = QLabel("──── 最近提问 ────")
        sep.setStyleSheet("color:#94a3b8;font-size:12px;")

        # 历史区固定高度（120px），内容超出时滚动。
        # 窗口总高不再随历史条数跳动，避免内容被遮挡/重绘残影。
        hist_wrap = QWidget()
        self._hist_box = QVBoxLayout(hist_wrap)
        self._hist_box.setContentsMargins(0, 0, 0, 0)
        self._hist_box.setSpacing(2)
        self._hist_area = QScrollArea()
        self._hist_area.setWidgetResizable(True)
        self._hist_area.setWidget(hist_wrap)
        self._hist_area.setFrameShape(QFrame.Shape.NoFrame)
        self._hist_area.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:#1e293b;border-radius:4px;}"
            "QScrollBar::handle:vertical{background:#64748b;border-radius:4px;min-height:24px;}"
            "QScrollBar::add-line:vertical{height:0;}QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}")
        self._hist_area.setFixedHeight(120)
        self._hist_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        body = QVBoxLayout(self)
        body.setContentsMargins(16, 10, 16, 12)
        body.setSpacing(6)
        body.addLayout(head)
        body.addWidget(self._scroll)
        body.addWidget(sep)
        body.addWidget(self._hist_area)
        body.addStretch(1)

        # 纯色背景，不透明，字才清楚
        self.setStyleSheet(
            "QWidget{background:#111827;border:1px solid #374151;border-radius:8px;}")

        # 系统级键盘钩子（不依赖窗口焦点/可见性）：演讲时焦点在 PPT 上也能触发。
        self._setup_global_hotkeys()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(200)

    def _setup_global_hotkeys(self):
        """keyboard 库全局热键：Esc=隐藏/显示, F9=暂停/恢复, Ctrl+Q=退出。
        回调发生在钩子线程，用 QueuedConnection 切回主线程操作 GUI（线程安全）。
        """
        try:
            import keyboard
            conn = Qt.ConnectionType.QueuedConnection
            keyboard.add_hotkey("esc", lambda: QMetaObject.invokeMethod(
                self, "toggle_visible", conn))
            keyboard.add_hotkey("f9", lambda: QMetaObject.invokeMethod(
                self, "toggle_pause", conn))
            keyboard.add_hotkey("ctrl+q", lambda: QMetaObject.invokeMethod(
                self, "quit_app", conn))
            log("全局热键已注册(keyboard): Esc=隐藏/显示, F9=暂停/恢复, Ctrl+Q=退出")
        except Exception as e:
            log(f"全局热键注册失败(不影响使用): {e}")

    @pyqtSlot()
    def toggle_visible(self):
        self.setVisible(not self.isVisible())
        log("窗口已隐藏（按 Esc 可重新显示）" if not self.isVisible()
            else "窗口已显示")

    def set_capture(self, cap):
        self._cap = cap

    @pyqtSlot()
    def quit_app(self):
        QApplication.instance().quit()

    @pyqtSlot()
    def toggle_pause(self):
        """F9：暂停/恢复监听。回答问题时暂停，避免自己的回答被识别成新问题。"""
        if self._cap is None:
            return
        paused = not self._cap._paused
        self._cap.set_paused(paused)
        if paused:
            self._status.setText("⏸ 已暂停（按 F9 恢复）")
            self._status.setStyleSheet("color:#f59e0b;font-size:12px;")
        else:
            self._status.setText("● 运行中")
            self._status.setStyleSheet("color:#22c55e;font-size:12px;")

    def _refresh_history(self):
        """主区域显示最新一句，历史区显示之前几句（小字）。"""
        while self._hist_box.count():
            it = self._hist_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for raw, fixed, zh, zh_q in reversed(self._history[:-1]):
            lab = QLabel(f"· {fixed}  →  {zh}")
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#cbd5e1;font-size:12px;")
            self._hist_box.addWidget(lab)
        self._hist_box.addStretch(1)
        # 历史区自动滚到底，显示最近一条历史
        hbar = self._hist_area.verticalScrollBar()
        QTimer.singleShot(50, lambda b=hbar: b.setValue(b.maximum()))

    def _poll(self):
        try:
            while True:
                ru, fixed, zh, zh_q = TEXT_Q.get_nowait()
                if ru == "*busy*":
                    self._raw.setText("")
                    self._ru.setText("识别中…")
                    self._zh.setText("（正在识别并生成回答，请稍候）")
                    continue
                if ru == "*idle*":
                    self._raw.setText("")
                    self._ru.setText("没听清，请再说一遍")
                    self._zh.setText("（识别为空：声音太小或太吵，可把麦克风音量调大）")
                    continue
                if ru == "*question*":
                    # 问题已识别并翻译，先显示；回答还在生成中
                    self._raw.setText(f"识别原文：{fixed}\n问题中文：{zh_q}"
                                      if zh_q else f"识别原文：{fixed}")
                    self._ru.setText("正在生成回答…")
                    self._zh.setText("（结合讲稿材料生成中，约 5~15 秒）")
                    continue
                self._history.append((ru, fixed, zh, zh_q))
                if len(self._history) > 3:
                    self._history.pop(0)
                if zh_q:
                    self._raw.setText(f"识别原文：{ru}\n问题中文：{zh_q}")
                else:
                    self._raw.setText("识别原文：" + ru)
                self._ru.setText(fixed)
                self._zh.setText("中文理解：" + zh)
                self._refresh_history()
                # 内容更新后自动滚到底部，保证最新文本立即可见
                bar = self._scroll.verticalScrollBar()
                QTimer.singleShot(50, lambda b=bar: b.setValue(b.maximum()))
                QTimer.singleShot(150, self.update)
        except queue.Empty:
            pass

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)


MUTEX_NAME = "PresenterCopilot_Overlay_SingleInstance"


def ensure_single_instance():
    """Windows 命名互斥体：已有实例在跑则返回 False（重复启动直接退出）。"""
    try:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        # 保存句柄引用，防止被 GC 释放导致互斥体失效
        _globals = globals()
        _globals["_mutex_handle"] = handle
        return True
    except Exception:
        return True  # 获取失败时不阻塞启动


def main():
    log("==== 启动演讲助手 ====")
    if not ensure_single_instance():
        print("已有演讲助手在运行，本次启动退出（如需重启请先退出旧实例）。")
        log("检测到已有实例，本次启动退出")
        return
    load_kb()
    load_materials()
    if KEYWORDS:
        log(f"已加载关键词引导: {KEYWORDS[:60]}...")
    app = QApplication(sys.argv)

    # 先加载模型，再显示窗口：避免加载期间出现"黑色未绘制窗口 + 沙漏卡死"
    print("加载 faster-whisper large-v3 ...")
    log("加载 faster-whisper large-v3 ...")
    dev, ctype = whisper_device()
    print(f"设备: {dev} / {ctype}")
    log(f"whisper 设备: {dev} / {ctype}")
    whisper = WhisperModel("large-v3", device=dev, compute_type=ctype)
    print("加载 NLLB 离线翻译 ...")
    log("加载 NLLB 离线翻译 ...")
    translator = OfflineTranslator()
    print("就绪。现在对麦克风说俄语，或让同学用俄语提问。")
    log("全部就绪，等待提问")

    overlay = Overlay()
    overlay.show()

    transcribe_q = queue.Queue()
    loopback = "--loopback" in sys.argv
    if loopback:
        log("模式: 系统内录（捕获电脑正在播放的声音）")
    cap = AudioCapture(transcribe_q, loopback=loopback,
                       save_dir=os.path.join(BASE_DIR, "audio"))
    overlay.set_capture(cap)
    cap.start()
    TranscribeWorker(transcribe_q, whisper, translator).start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
