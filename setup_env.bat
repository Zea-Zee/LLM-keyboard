@echo off
echo === LLM-Keyboard Setup ===
echo Создание conda окружения llmkeyboard...
call conda create -n llmkeyboard python=3.11 -y
call conda activate llmkeyboard
echo Установка PyTorch с CUDA 12.1...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
echo Установка зависимостей...
call pip install -r requirements.txt
echo Установка Unsloth (для обучения)...
call pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
echo.
echo === Готово! ===
echo Для обучения нужен CUDA toolkit 12.1+
echo.
echo Запуск клавиатуры: conda activate llmkeyboard ^&^& python main.py --keyboard
echo Запуск overlay:    conda activate llmkeyboard ^&^& python main.py --overlay
pause
