#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка локальных моделей распознавания речи (Vosk, Apache 2.0).

Модели в репозиторий не кладём — они весят слишком много. Этот скрипт кладёт их
в models/ один раз. После него проект работает полностью офлайн.

    python scripts/get_models.py          # компактный набор, ~180 МБ
    python scripts/get_models.py --full   # с большой казахской моделью, ~1.5 ГБ
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BASE = "https://alphacephei.com/vosk/models/"

# Имя -> (описание, входит ли в компактный набор)
CATALOG = [
    ("vosk-model-small-ru-0.22", "русский, компактная", True),
    ("vosk-model-small-kz-0.42", "казахский, компактная", True),
    ("vosk-model-spk-0.4", "различение говорящих по голосу", True),
    ("vosk-model-kz-0.42", "казахский, большая (точнее, 1.3 ГБ)", False),
    ("vosk-model-ru-0.42", "русский, большая (точнее, 1.8 ГБ)", False),
]


def download(name):
    target = MODELS / name
    if target.exists():
        print("  уже есть: %s" % name)
        return
    archive = MODELS / (name + ".zip")
    url = BASE + name + ".zip"
    print("  качаю %s ..." % name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(MODELS)
    finally:
        if archive.exists():
            archive.unlink()
    print("  готово: %s" % name)


def main():
    parser = argparse.ArgumentParser(description="Загрузка локальных моделей Vosk")
    parser.add_argument("--full", action="store_true",
                        help="добавить большие модели: точнее, но дольше качать")
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    wanted = [(n, d) for n, d, small in CATALOG if small or args.full]
    print("Загружаю %d моделей в %s" % (len(wanted), MODELS))
    for name, description in wanted:
        print("- %s (%s)" % (name, description))
        download(name)

    print("\nГотово. Локальные модели на месте:")
    for path in sorted(MODELS.glob("vosk-model-*")):
        print("  %s" % path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
