# -*- coding: utf-8 -*-
"""Qwen 纠错 + 翻译层：把 whisper 识别结果修正为通顺俄语并翻译成中文。

依赖：Ollama 服务运行中（localhost:11434），模型 qwen2.5:7b。
失败时返回 (None, None)，由调用方降级到 NLLB 翻译。
"""
import json
import re
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
TIMEOUT = 90


def _ask(prompt, timeout=TIMEOUT):
    # num_predict 限制输出长度（防冗长拖慢），keep_alive 30 分钟（防答辩中途模型被卸载重载）
    body = json.dumps({"model": MODEL, "prompt": prompt,
                       "stream": False, "temperature": 0.1,
                       "num_predict": 350, "keep_alive": "30m"}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]


def fix(ru_text, context):
    """context: 演讲主题背景（俄语）。返回 (修正俄语, 中文翻译)；失败返回 (None, None)。"""
    system = (
        "背景：{ctx}\n"
        "任务：下面是一段俄语语音识别文本，可能有个别单词识别错误。"
        "请结合上述演讲主题，把它修正为正确、通顺的俄语（保持原意），然后给出准确的中文翻译。"
        "严格按以下格式输出：\n"
        "修正俄语：<修正后的俄语>\n中文翻译：<中文翻译>"
    ).format(ctx=context)
    try:
        out = _ask(f"{system}\n\n识别文本：{ru_text}")
    except Exception:
        return None, None
    fixed = re.search(r"修正俄语[:：]\s*(.+)", out)
    zh = re.search(r"中文翻译[:：]\s*(.+)", out)
    if not fixed or not zh:
        return None, None
    return fixed.group(1).strip(), zh.group(1).strip()


def answer(ru_text, context, materials, timeout=120):
    """结合材料生成回答。materials: [{'text': ...}, ...]。返回 (俄语回答, 中文理解)；失败 (None, None)。"""
    mats = "\n".join(f"[{i}] {b['text']}" for i, b in enumerate(materials))
    system = (
        "背景：{ctx}\n"
        "以下是演讲者的讲稿材料片段（与提问相关的部分）：\n"
        "{mats}\n"
        "老师用俄语提问，内容来自语音识别，可能有单词错误，请结合材料和语境推断真实意图。\n"
        "任务：用俄语给出完整、专业、口语化的回答（3-5 句，演讲者可直接照读），然后给出中文翻译。\n"
        "严格按以下格式输出：\n"
        "俄语回答：<回答>\n中文理解：<中文翻译>"
    ).format(ctx=context, mats=mats)
    try:
        out = _ask(f"{system}\n\n老师提问：{ru_text}", timeout=timeout)
    except Exception:
        return None, None
    ru_ans = re.search(r"俄语回答[:：]\s*(.+)", out)
    zh_ans = re.search(r"中文理解[:：]\s*(.+)", out)
    if not ru_ans or not zh_ans:
        return None, None
    return ru_ans.group(1).strip(), zh_ans.group(1).strip()
