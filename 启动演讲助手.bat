@echo off
rem Presenter Copilot: start overlay window (no console)
rem 提示：把下面的模型缓存路径改成你自己的 D 盘路径（避免占用 C 盘）
set HF_HUB_CACHE=D:\ai\models\hf
set HF_HOME=D:\ai\models\hf
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set OLLAMA_MODELS=D:\ollama\models
set PATH=%PATH%;%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin;%~dp0.venv\Lib\site-packages\nvidia\cublas\bin
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul || start "" "ollama.exe" serve
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0m1_stt_overlay.py"
