# LLM-Keyboard

AI-ассистент для Windows 11, предоставляющий предиктивный ввод (Next Word Prediction) на базе локальной LLM (Qwen 3.5 0.8B) с поддержкой персональных LoRA-адаптеров.

## Особенности
- **Strictly Windows Native:** Без WSL, без C++ зависимостей (flash-attention не требуется).
- **Sampling + Majority Voting:** 10 параллельных сэмплов → топ-5 слов по голосованию.
- **Adapter Stacking:** Замороженный CPT-адаптер (домен Telegram) + обучаемый персональный адаптер (стиль пользователя). Три переключаемых режима: Base / CPT / Personal.
- **Mobile Keyboard UI:** PyQt6-окно с RU/EN/цифры/символы раскладками, строка подсказок с адаптивным шрифтом, поле ввода.
- **Global Overlay:** Строка подсказок поверх всех окон. Хоткей Ctrl+Shift+Space для включения/выключения.
- **Personal Training:** Обучение персонального LoRA-адаптера на основе экспорта истории Telegram.

## Требования
- Windows 11
- NVIDIA GPU с CUDA 12.1+ (рекомендуется для скорости; CPU тоже работает, но медленнее)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) или Anaconda

## Установка

```bat
git clone ...
cd LLM-keyboard
setup_env.bat
```

## Запуск

```bat
conda activate llmkeyboard

# Режим мобильной клавиатуры (демо/тестирование)
python main.py --keyboard

# Режим глобального overlay (рабочий режим)
python main.py --overlay
```

## Настройка адаптеров

Адаптеры настраиваются через UI Settings (кнопка ⚙):
- **CPT Адаптер:** HuggingFace Hub (repo_id + revision) или локальная папка
- **Персональный Адаптер:** аналогично; обучается прямо в приложении

## Обучение персонального адаптера

1. Экспортировать историю Telegram в HTML (`Настройки → Экспорт данных → HTML`)
2. В приложении: `Обучение → Выбрать папку экспорта → Указать Telegram User ID`
3. Нажать `Обработать`, дождаться подготовки датасета
4. Нажать `Запустить обучение`

## Структура проекта

```
core/         Inference engine, конфиг, буфер ввода
ui/           PyQt6 окна (клавиатура, overlay, настройки, обучение)
training/     Обёртка unsloth, обработка данных
parsing/      Парсер HTML экспорта Telegram
main.py       Точка входа
CURSOR.md     Документация + TODO для AI-агентов
```
