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
# Сколько секунд с начала записи слушаем, чтобы определить язык.
PROBE_SECONDS = float(os.environ.get("PROBE_SECONDS", "45"))
# Смешанная речь: разбираем запись обеими моделями и подставляем казахские слова
# в русскую основу. STT_MIXED=0 выключает и экономит примерно половину времени.
MIXED_MARGIN = float(os.environ.get("MIXED_MARGIN", "0.05"))
CHUNK = 8000


def _utterance(words, spk, lang=None):
    confs = [w.get("conf", 0.0) for w in words]
    return {
        "start": round(words[0]["start"], 2),
        "end": round(words[-1]["end"], 2),
        "text": " ".join(w["word"] for w in words),
        "words": words,  # нужны, чтобы разрезать реплику по смене говорящего
        "conf": sum(confs) / len(confs) if confs else 0.0,
        "lang": lang,
        "spk": spk,
    }


def _overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


# Буквы, которых в русском алфавите нет вовсе. Русская модель физически не может
# их выдать, поэтому их появление — надёжный признак казахской речи. Уверенность
# для этого не годится: Vosk уверенно ошибается на чужом языке и выдал
# «жара и до семин» вместо «жарайды келісемін» с высоким баллом.
KAZAKH_LETTERS = set("әғқңөұүһі")


def _kazakh_letters(text):
    return sum(1 for ch in (text or "").lower() if ch in KAZAKH_LETTERS)


def _best_rival(word, foreign, used):
    """Ближайшее по времени казахское слово, перекрывающее это не меньше чем наполовину."""
    span = max(word["end"] - word["start"], 0.01)
    best, best_share = None, 0.5
    for candidate in foreign:
        if id(candidate) in used:
            continue
        overlap = min(word["end"], candidate["end"]) - max(word["start"], candidate["start"])
        share = overlap / span
        if share > best_share:
            best, best_share = candidate, share
    return best


def merge_by_confidence(primary, secondary, margin=None):
    """Смешанная речь: подставляем казахские слова в русскую стенограмму.

    Совещания здесь двуязычные, и одна модель на всю запись теряет вторую
    половину. Сливаем на уровне слов, а не реплик: русская модель даёт основу,
    а там, где казахская услышала слово с казахскими буквами, встаёт её вариант.
    Заменять реплику целиком нельзя — вместе с казахской фразой портится
    окружающий русский текст.
    """
    margin = MIXED_MARGIN if margin is None else margin
    foreign = [word for utt in secondary for word in utt.get("words", [])
               if _kazakh_letters(word.get("word"))]
    if not foreign:
        return primary, 0

    replaced, used = 0, set()
    for utt in primary:
        words = []
        for word in utt.get("words", []):
            rival = _best_rival(word, foreign, used)
            # Казахская модель выдумывает казахские слова и на русской речи:
            # на нашей записи она услышала «сияқты» там, где сказано «сейчас».
            # Поэтому мало найти слово с казахскими буквами — оно должно быть
            # ещё и увереннее того, что услышала русская модель.
            if rival is not None and rival.get("conf", 0.0) - word.get("conf", 0.0) > margin:
                used.add(id(rival))
                words.append(rival)
                replaced += 1
            else:
                words.append(word)
        utt["words"] = words
        utt["text"] = " ".join(w["word"] for w in words)
    return primary, replaced


_model_cache = {}


class LocalSttError(RuntimeError):
    pass


def _scan():
    """Все языковые модели на диске: {язык: [пути]}."""
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
                found.setdefault(lang, []).append(path)
                break
    return found


def available_models():
    """Лучшая модель на каждый язык: большая точнее маленькой."""
    return {lang: sorted(paths, key=lambda p: "small" in p.name)[0]
            for lang, paths in _scan().items()}


def probe_models():
    """Самая лёгкая модель на язык — ей быстро определяем, на чём говорят."""
    return {lang: sorted(paths, key=lambda p: "small" not in p.name)[0]
            for lang, paths in _scan().items()}


def _load(path, cls=None):
    key = str(path)
    if key not in _model_cache:
        from vosk import Model, SetLogLevel, SpkModel
        SetLogLevel(-1)  # Vosk иначе заливает консоль отладкой Kaldi
        _model_cache[key] = (SpkModel if cls == "spk" else Model)(str(path))
    return _model_cache[key]


def warmup(log=print):
    """Заранее поднимаем модели в память.

    Большая модель грузится с диска около двух минут, и без прогрева это ждёт
    первый же пользователь. В сервере модели живут в памяти, поэтому платим
    один раз при старте, а каждый следующий разбор идёт втрое быстрее.
    """
    models = available_models()
    if not models:
        return
    for lang, path in sorted(models.items()):
        log("  загружаю %s (%s)" % (path.name, lang))
        _load(path)
    for lang, path in sorted(probe_models().items()):
        _load(path)
    spk = MODELS_DIR / "vosk-model-spk-0.4"
    if spk.exists():
        _load(spk, "spk")
    try:
        from meeting import diarize
        if diarize.available():
            log("  загружаю модели разделения на говорящих")
            diarize._build()
    except Exception as exc:
        log("  диаризация не поднялась: %s" % exc)
    log("  модели готовы")


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


def _recognize(wav_path, model_path, spk_model, limit_seconds=None, lang_tag=None):
    """Один проход одной моделью. Возвращает реплики и среднюю уверенность.

    limit_seconds ограничивает разбор началом записи — этого достаточно, чтобы
    понять язык, и не нужно гонять всю запись дважды.
    """
    from vosk import KaldiRecognizer

    audio = wave.open(str(wav_path), "rb")
    recognizer = KaldiRecognizer(_load(model_path), audio.getframerate())
    recognizer.SetWords(True)
    if spk_model is not None:
        recognizer.SetSpkModel(spk_model)

    frame_budget = int(limit_seconds * audio.getframerate()) if limit_seconds else None
    read_frames = 0
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
                utterances.append(_utterance(chunk, result.get("spk"), lang_tag))
                chunk = []
            chunk.append(word)
        if chunk:
            utterances.append(_utterance(chunk, result.get("spk"), lang_tag))

    while True:
        data = audio.readframes(CHUNK)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            collect(recognizer.Result())
        read_frames += CHUNK
        if frame_budget and read_frames >= frame_budget:
            break
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
        # Язык определяем по началу записи лёгкими моделями, а полный проход
        # делаем один раз и только победившей. Иначе большие модели гоняют всю
        # запись дважды, и на двух минутах это четыре минуты ожидания.
        forced = os.environ.get("STT_LANGUAGE", "").strip().lower()
        probe, mixed, replaced = [], False, 0
        if forced in models:
            lang = forced
        elif len(models) == 1:
            lang = next(iter(models))
        else:
            for probe_lang, probe_path in probe_models().items():
                _, probe_conf = _recognize(wav_path, probe_path, None, PROBE_SECONDS, probe_lang)
                probe.append({"language": probe_lang, "model": probe_path.name,
                              "confidence": round(probe_conf, 3)})
            probe.sort(key=lambda p: p["confidence"], reverse=True)
            lang = probe[0]["language"]
            # Разрыв уверенностей на пробе казался хорошим признаком смешанной
            # речи, но оказался бесполезным: на записи с казахской фразой он был
            # 0.125, а на почти полностью русской записи из задания — 0.016.
            # Поэтому на смешанную речь идём всегда, когда есть обе модели:
            # пункт 4 задания важнее лишней минуты разбора.
            mixed = len(probe) > 1 and os.environ.get("STT_MIXED", "1") != "0"

        model_path = models[lang]
        utterances, confidence = _recognize(wav_path, model_path, spk_model, None, lang)
        model_name = model_path.name

        if mixed:
            other = probe[1]["language"]
            rival, _ = _recognize(wav_path, models[other], spk_model, None, other)
            utterances, replaced = merge_by_confidence(utterances, rival)
            model_name = "%s + %s" % (model_path.name, models[other].name)

        # Отдельная модель диаризации размечает речь точнее, чем вектор голоса
        # на целую реплику Vosk: она видит смену говорящего внутри реплики.
        from meeting import diarize
        diarization = "vosk-xvector"
        if diarize.available():
            try:
                segments = diarize.split_by_turns(utterances, diarize.turns(wav_path))
                diarization = "pyannote+titanet"
            except Exception as exc:  # не смогли — остаёмся на запасном способе
                print("  диаризация недоступна, работаем по векторам Vosk: %s" % exc)
                segments = None
        else:
            segments = None
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if segments is None:
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
            "diarization": diarization,
            "speakers": len({s["speaker"] for s in segments}),
            "confidence": round(confidence, 3),
            "language_probe": probe,
            "mixed_speech": mixed,
            "replaced_utterances": replaced,
        },
    }
