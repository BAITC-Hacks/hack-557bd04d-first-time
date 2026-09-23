"""Разделение записи на говорящих — отдельной моделью, локально.

Почему не хватало прежнего способа: Vosk отдаёт один вектор голоса на целую
реплику, а реплика у него кончается на длинной паузе. Если в неё попали двое
(а в живом совещании перебивают постоянно), вектор получается усреднённым и
человек определяется неверно.

Здесь работает специализированный конвейер: pyannote размечает, где вообще есть
речь и где происходит смена говорящего, а titanet считает эмбеддинг голоса для
каждого такого куска. Кластеризация по эмбеддингам даёт реплики говорящих с
точными границами — и уже к ним привязываются распознанные слова.

Обе модели в формате ONNX работают офлайн через sherpa-onnx.
"""

import os
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "diarization"
SEGMENTATION = MODELS_DIR / "segmentation.onnx"
EMBEDDING = MODELS_DIR / "embedding.onnx"

# Порог кластеризации: ниже — охотнее объединяет голоса в одного человека.
CLUSTER_THRESHOLD = float(os.environ.get("DIARIZATION_THRESHOLD", "0.70"))
# Сколько говорящих ожидать. 0 — определить самостоятельно.
NUM_SPEAKERS = int(os.environ.get("DIARIZATION_SPEAKERS", "0"))

_pipeline = None


def available():
    return SEGMENTATION.exists() and EMBEDDING.exists()


def _build():
    """Собираем конвейер один раз: загрузка моделей заметно дороже прогона."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION)),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=NUM_SPEAKERS if NUM_SPEAKERS > 0 else -1,
            threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3,   # реплику короче трети секунды не считаем речью
        min_duration_off=0.5,  # пауза короче полсекунды не разрывает реплику
    )
    if not config.validate():
        raise RuntimeError("sherpa-onnx отверг конфигурацию диаризации")
    _pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _pipeline


def turns(wav_path):
    """Возвращает [(начало, конец, номер говорящего)] по всей записи."""
    import numpy as np

    pipeline = _build()
    with wave.open(str(wav_path), "rb") as audio:
        if audio.getframerate() != pipeline.sample_rate:
            raise RuntimeError("ожидается {} Гц".format(pipeline.sample_rate))
        raw = audio.readframes(audio.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    result = pipeline.process(samples).sort_by_start_time()
    return [(seg.start, seg.end, seg.speaker) for seg in result]


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def apply(segments, speaker_turns):
    """Проставляем говорящего каждой реплике по наибольшему пересечению по времени."""
    if not speaker_turns:
        return segments
    for segment in segments:
        best, best_overlap = None, 0.0
        for start, end, speaker in speaker_turns:
            shared = _overlap(segment["start"], segment["end"], start, end)
            if shared > best_overlap:
                best, best_overlap = speaker, shared
        if best is not None:
            segment["speaker"] = "Спикер %d" % (best + 1)
    return segments


def split_by_turns(segments, speaker_turns):
    """Режем реплику, если внутри неё сменился говорящий.

    Главный выигрыш против прежнего способа: слова одной длинной реплики Vosk
    расходятся по разным людям, а не приписываются одному.
    """
    if not speaker_turns:
        return segments
    # Кластеры приходят с произвольными номерами (0, 1, 3) — перенумеровываем
    # подряд в порядке появления, иначе в интерфейсе видно «Спикер 1, 2, 4».
    order = {}
    for _start, _end, who in speaker_turns:
        order.setdefault(who, len(order) + 1)

    result = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            result.append(segment)
            continue
        current, chunk = None, []
        for word in words:
            middle = (word["start"] + word["end"]) / 2
            speaker = None
            for start, end, who in speaker_turns:
                if start <= middle <= end:
                    speaker = who
                    break
            if speaker is None:
                speaker = current
            if current is not None and speaker != current and chunk:
                result.append(_pack(chunk, order.get(current, 1)))
                chunk = []
            current = speaker
            chunk.append(word)
        if chunk:
            result.append(_pack(chunk, order.get(current, 1)))
    return result


def _pack(words, number):
    return {
        "start": round(words[0]["start"], 2),
        "end": round(words[-1]["end"], 2),
        "speaker": "Спикер %d" % number,
        "text": " ".join(w["word"] for w in words),
    }
