# Presenter Copilot · 离线答辩问答助手

**在 PPT 答辩 / 演讲现场，听清老师的提问，结合你的讲稿材料，即时生成"俄语回答（可照读）+ 中文理解（本机看）"，全程离线，仅在你电脑显示。**

## 为什么做这个

- 你在国外用非母语演讲（本项目面向俄语答辩场景），现场可能听不清老师的问题
- 答辩教室网络通常很差，云端语音识别 / AI 服务不可用
- 你的讲稿、论文是私有材料，不希望上传到任何服务器

本项目把 **语音识别 + 翻译 + 结合材料生成回答** 全部跑在本地：离线可用、数据不出本机、投影仪不可见。

## 功能

| 能力 | 说明 |
| --- | --- |
| 俄语提问识别 | faster-whisper large-v3 本地 GPU 识别，带关键词引导（读你的 keywords.txt） |
| 问题即时翻译 | NLLB-200 离线翻译，识别后约 1 秒显示问题中文 |
| 结合材料回答 | 本地检索你的讲稿/论文切片（RAG），Qwen2.5:7b 生成"俄语回答 + 中文理解" |
| 完全离线 | 识别、翻译、回答全部本地，断网可用；在线识别仅作可选补充 |
| 隐私 | 材料、录音、问答记录全部留在本机，不联网上传 |
| 悬浮窗置顶 | 无边框置顶小窗，与 PPT 演讲者视图同屏，投影仪不可见 |
| 防回声死循环 | 你自己读回答时不会被识别成新问题 |
| 防幻觉 | 无语音片段 / 幻觉模板自动拦截 |

## 环境要求

- Windows 10/11，NVIDIA GPU（推荐 8GB 显存，可跑 large-v3 + Qwen 7B）
- Python 3.12
- [Ollama](https://ollama.com)（本地运行 Qwen）

> 纯 CPU 也能跑，但识别与回答速度会明显变慢。

## 安装

```bat
:: 1. 创建虚拟环境（务必装到非 C 盘，模型很大）
py -3.12 -m venv D:\presenter-copilot\.venv

:: 2. 安装依赖
D:\presenter-copilot\.venv\Scripts\pip install -r requirements.txt

:: 3. 安装 Ollama 并拉取 Qwen（约 4.7GB）
ollama pull qwen2.5:7b

:: 4. 设置模型缓存到 D 盘（避免占用 C 盘）
setx HF_HUB_CACHE D:\ai\models\hf
setx HF_HOME D:\ai\models\hf
setx OLLAMA_MODELS D:\ollama\models
```

首次启动时 faster-whisper（large-v3）与 NLLB-200 模型会自动下载到上述缓存目录（各约 3-6GB，只下载一次）。

> ⚠️ 若用 NVIDIA GPU，需要保证 PATH 含 `cudnn/bin` 与 `cublas/bin`（通常位于 `.venv\Lib\site-packages\nvidia\...`）。`启动演讲助手.bat` 已处理；手动启动需自行追加。

## 使用

1. 把你的讲稿 / 论文（`.pptx/.docx/.pdf/.txt`）放进 `materials/` 目录，双击 `导入材料.bat`（材料更新后重跑一次）
2. 双击 `启动演讲助手.bat` 启动悬浮窗（会自动拉起 Ollama）
3. 老师提问时正常听即可——识别到问题后先显示"问题 + 中文翻译"，随后生成"俄语回答 + 中文理解"
4. 按自己的节奏照读俄语回答即可（防回声过滤已开启）

### 快捷键

| 按键 | 功能 |
| --- | --- |
| `Esc` | 隐藏 / 显示悬浮窗 |
| `F9` | 暂停 / 恢复监听 |
| `Ctrl+Q` | 退出 |

窗口右上角：`—` 最小化（进任务栏）、`×` 退出。

### 内录模式（线上答辩 / 播放提问音频）

```bat
pythonw m1_stt_overlay.py --loopback
```

捕获电脑正在播放的声音，适合线上答辩或测试时播放提问音频。

## 目录结构

```
presenter-copilot/
├── m1_stt_overlay.py      # 主程序（采集+识别+翻译+回答+悬浮窗）
├── qwen_fix.py            # Qwen 生成回答 / 纠错
├── nllb_translate.py      # NLLB 离线翻译兜底
├── import_materials.py    # 材料导入（切片入库）
├── requirements.txt
├── keywords.txt           # 俄语识别关键词引导（按你的讲稿修改）
├── context.txt            # 演讲背景（Qwen 纠错上下文）
├── 启动演讲助手.bat
└── 导入材料.bat
```

运行时生成：`materials_data.json`（材料切片库）、`audio/`（最近 60 段录音）、`qa_history.md`（问答记录）、`app.log`。

## 依赖与许可证

| 组件 | 许可证 |
| --- | --- |
| faster-whisper | MIT |
| Qwen2.5 模型（Ollama） | Apache-2.0 |
| Ollama | MIT |
| ctranslate2 | MIT |
| speech_recognition | BSD-3-Clause |
| PyQt6 | GPL-3.0 |
| NLLB-200 模型 | CC-BY-NC-4.0（非商用） |

> 本项目代码以 MIT 发布。注意：NLLB-200 模型权重为 **CC-BY-NC-4.0（非商用）**，PyQt6 为 **GPL-3.0**——若要做闭源商业产品，请分别替换为可商用翻译模型与 PySide6。

## 致谢与说明

- 功能设计参考了 [realtime-call-translator](https://github.com/AsaAsasa/realtime-call-translator)、[xexamai](https://github.com/Artasov/xexamai)、CampusMate 等项目的思路（悬浮窗、热键、识别-回答链路等通用功能）
- **全部代码为原创实现**，针对俄语答辩场景定制（关键词引导、幻觉拦截、回声过滤、材料检索打分、双语照读格式等）
- 语音识别：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（OpenAI Whisper 的优化实现）
- 本地 LLM：[Ollama](https://ollama.com) + Qwen2.5

## 隐私

- 所有处理在本机完成：识别、翻译、回答生成均不联网
