"""Локальное распознавание речи: Vosk + ffmpeg. Ни байта наружу.

ТЗ запрещает передавать аудио и текст во внешние облачные API и требует, чтобы
решение можно было развернуть в закрытом контуре заказчика. Поэтому распознавание
по умолчанию идёт здесь, на машине: модели Vosk (Apache 2.0) лежат в models/.

Язык заранее не угадываем. Совещания в Казахстане — это чаще всего смесь русского
и казахского, и ошибиться с выбором модели дорого. Поэтому запускаем обе локальные
модели и оставляем ту, в которой распознавание увереннее. На приложенных к заданию
записях так выигрывает русская модель, на казахской речи — казахская.

Говорящих различаем тоже локально: Vosk выдаёт вектор голоса (x-vector) на каждую
реплику, а мы группируем реплики по близости векторов.
"""

import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Порог схожести голосов: выше — та же реплика того же человека.
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.62"))
CHUNK = 8000

_model_cache = {}


class LocalSttError(RuntimeError):
    pass


def available_models():
    """Какие языковые модели реально лежат на диске."""
    if not MODELS_DIR.exists():
        return {}
    found = {}
    for path in sorted(MODELS_DIR.iterdir()):
        name = path.name
        if not path.is_dir() or not name.startswith("vosk-model-") or "-spk-" in name:
            continue
        # vosk-model-small-ru-0.22 -> ru, vosk-model-kz-0.42 -> kz
        for lang in ("ru", "kk", "kz", "en"):
            if "-%s-" % lang in name:
                # большая модель лучше маленькой — она и остаётся
                if lang not in found or "small" in found[lang].name:
                    found[lang] = path
                break
    return found


def _load(path, cls=None):
    key = str(path)
    if key not in _model_cache:
        from vosk import Model, SetLogLevel, SpkModel
        SetLogLevel(-1)  # Vosk иначе заливает консоль отладкой Kaldi
        _model_cache[key] = (SpkModel if cls == "spk" else Model)(str(path))
    return _model_cache[key]


def _ffmpeg():
    """ffmpeg приезжает пакетом imageio-ffmpeg — отдельная установка не нужна."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def to_wav(src):
    """Любой аудио- или видеофайл -> WAV 16 кГц моно, как ждёт Vosk."""
    dst = Path(tempfile.gettempdir()) / ("stt_%s.wav" % os.getpid())
    proc = subprocess.run(
        [_ffmpeg(), "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "wav", str(dst)],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0 or not dst.exists():
        raise LocalSttError("ffmpeg не смог прочитать файл: " + (proc.stderr or "")[-300:])
    return dst


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _assign_speakers(utterances):
    """Группируем реплики по голосу: близкий вектор — тот же человек."""
    centroids = []
    for utt in utterances:
        vec = utt.get("spk")
        if not vec:
            utt["speaker"] = "Спикер 1" if not centroids else utt.get("speaker", "Спикер 1")
            continue
        best, best_score = -1, SPEAKER_THRESHOLD
        for index, (centroid, count) in enumerate(centroids):
            score = _cosine(vec, centroid)
            if score > best_score:
                best, best_score = index, score
        if best < 0:
            centroids.append((list(vec), 1))
            best = len(centroids) - 1
        else:  # усредняем центроид, чтобы он не «уплывал» от одной реплики
            centroid, count = centroids[best]
            centroids[best] = ([(c * count + v) / (count + 1) for c, v in zip(centroid, vec)], count + 1)
        utt["speaker"] = "Спикер %d" % (best + 1)
    return utterances


def _recognize(wav_path, model_path, spk_model):
    """Один проход одной моделью. Возвращает реплики и среднюю уверенность."""
    from vosk import KaldiRecognizer

    audio = wave.open(str(wav_path), "rb")
    recognizer = KaldiRecognizer(_load(model_path), audio.getframerate())
    recognizer.SetWords(True)
    if spk_model is not None:
        recognizer.SetSpkModel(spk_model)

    utterances, confidences = [], []

    def collect(raw):
        result = json.loads(raw)
        words = result.get("result") or []
        text = (result.get("text") or "").strip()
        if not text or not words:
            return
        confidences.extend(w.get("conf", 0.0) for w in words)
        utterances.append({
            "start": round(words[0]["start"], 2),
            "end": round(words[-1]["end"], 2),
            "text": text,
            "spk": result.get("spk"),
        })

    while True:
        data = audio.readframes(CHUNK)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            collect(recognizer.Result())
    collect(recognizer.FinalResult())
    audio.close()

    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return utterances, confidence


def transcribe(audio_path, title=None):
    """Полное локальное распознавание. Возвращает тот же формат, что облачный путь."""
    models = available_models()
    if not models:
        raise LocalSttError(
            "Нет локальных моделей в models/. Скачайте их: python scripts/get_models.py")

    spk_path = MODELS_DIR / "vosk-model-spk-0.4"
    spk_model = _load(spk_path, "spk") if spk_path.exists() else None

    wav_path = to_wav(audio_path)
    try:
        attempts = []
        for lang, path in models.items():
            utterances, confidence = _recognize(wav_path, path, spk_model)
            attempts.append((confidence, lang, path.name, utterances))
        attempts.sort(key=lambda a: a[0], reverse=True)
        confidence, lang, model_name, utterances = attempts[0]
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    segments = [
        {"start": u["start"], "end": u["end"], "speaker": u["speaker"], "text": u["text"]}
        for u in _assign_speakers(utterances)
    ]
    return {
        "title": title or Path(audio_path).stem,
        "duration_sec": round(segments[-1]["end"], 1) if segments else 0,
        "segments": segments,
        "transcription_mode": "local",
        "asr": {
            "model": model_name,
            "language": lang,
            "confidence": round(confidence, 3),
            "considered": [{"language": a[1], "model": a[2], "confidence": round(a[0], 3)}
                           for a in attempts],
        },
    }
