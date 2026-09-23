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


# Диаризация: сегментация речи pyannote + эмбеддинги голоса titanet, обе в ONNX.
DIARIZATION = MODELS / "diarization"
SEGMENTATION_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMBEDDING_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                 "speaker-recongition-models/nemo_en_titanet_small.onnx")


def fetch(url, target):
    with urllib.request.urlopen(url, timeout=180) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_diarization():
    """Модели разделения на говорящих: ~46 МБ, работают офлайн."""
    DIARIZATION.mkdir(parents=True, exist_ok=True)
    embedding = DIARIZATION / "embedding.onnx"
    segmentation = DIARIZATION / "segmentation.onnx"

    if embedding.exists():
        print("  уже есть: embedding.onnx")
    else:
        print("  качаю эмбеддинги голоса ...")
        fetch(EMBEDDING_URL, embedding)

    if segmentation.exists():
        print("  уже есть: segmentation.onnx")
        return
    print("  качаю сегментацию речи ...")
    archive = DIARIZATION / "segmentation.tar.bz2"
    try:
        fetch(SEGMENTATION_URL, archive)
        import tarfile
        with tarfile.open(archive, "r:bz2") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/model.onnx"))
            member.name = "segmentation.onnx"
            tar.extract(member, DIARIZATION)
    finally:
        if archive.exists():
            archive.unlink()
    print("  готово: segmentation.onnx")


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

    print("- разделение на говорящих (pyannote + titanet)")
    download_diarization()

    print("\nГотово. Локальные модели на месте:")
    for path in sorted(MODELS.glob("vosk-model-*")):
        print("  %s" % path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
