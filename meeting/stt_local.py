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
# Ниже этого слово почти наверняка не расслышано, а придумано декодером.
MIN_WORD_CONF = float(os.environ.get("MIN_WORD_CONF", "0.30"))
# Пауза между словами, после которой начинается новая реплика, секунды.
PAUSE_SPLIT = float(os.environ.get("PAUSE_SPLIT", "0.9"))
CHUNK = 8000


def _utterance(words, spk):
    return {
        "start": round(words[0]["start"], 2),
        "end": round(words[-1]["end"], 2),
        "text": " ".join(w["word"] for w in words),
        "spk": spk,
    }

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


# Фильтры ffmpeg перед распознаванием. По умолчанию выключены: на нашей тестовой
# записи с телефона подавление шума сделало хуже — уверенность упала с 0.860 до
# 0.844, а реплики склеились. Оставлено настройкой для заведомо шумных записей,
# разумное значение: highpass=f=80,dynaudnorm=f=150:g=15
AUDIO_FILTERS = os.environ.get("AUDIO_FILTERS", "")


def to_wav(src, clean=True):
    """Любой аудио- или видеофайл -> WAV 16 кГц моно, как ждёт Vosk."""
    dst = Path(tempfile.gettempdir()) / ("stt_%s.wav" % os.getpid())
    command = [_ffmpeg(), "-y", "-i", str(src)]
    if clean and AUDIO_FILTERS:
        command += ["-af", AUDIO_FILTERS]
    command += ["-ar", "16000", "-ac", "1", "-f", "wav", str(dst)]
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
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
        if not words:
            return
        confidences.extend(w.get("conf", 0.0) for w in words)
        # Слова с очень низкой уверенностью — это почти всегда шум, распознанный
        # как речь. В стенограмме они мешают читать, а модель разбора сбивают.
        kept = [w for w in words if w.get("conf", 0.0) >= MIN_WORD_CONF]
        if not kept:
            return
        # Vosk отдаёт реплику целиком до длинной паузы: на непрерывной речи это
        # простыня на полминуты. Режем по паузам между словами.
        chunk = [kept[0]]
        for word in kept[1:]:
            if word["start"] - chunk[-1]["end"] > PAUSE_SPLIT:
                utterances.append(_utterance(chunk, result.get("spk")))
                chunk = []
            chunk.append(word)
        if chunk:
            utterances.append(_utterance(chunk, result.get("spk")))

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
