"""Конвейер: аудио -> транскрипт -> протокол и поручения.

Два режима работы:
  * боевой — Soniox распознаёт речь, Claude извлекает поручения;
  * демо — берётся готовый транскрипт из samples/, ключи не нужны.

Демо-режим существует не для красоты: проверяющий должен запустить проект
без наших личных аккаунтов, и демонстрация не должна зависеть от сети в зале.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

SONIOX_BASE = "https://api.soniox.com"
POLL_SECONDS = 3
POLL_ATTEMPTS = 200

DEFAULT_MODEL = "claude-opus-5"


def load_env():
    """Читаем .env рядом с проектом. Ключи не должны жить в коде и в git."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def stt_provider():
    """Кто распознаёт речь. По умолчанию локально — этого требует ТЗ."""
    choice = os.environ.get("STT_PROVIDER", "local").strip().lower()
    if choice == "local":
        from meeting import stt_local
        return "local" if stt_local.available_models() else "demo"
    if choice == "soniox" and os.environ.get("SONIOX_API_KEY"):
        return "soniox"
    return "demo"


def analysis_provider():
    """Кто разбирает стенограмму: локальная модель, облачная или заглушка.

    Явная настройка важнее всего — иначе контрольный сценарий из README давал бы
    разный результат на разных машинах. Без настройки ключ Anthropic считаем
    осознанным выбором облака, иначе берём локальную модель, если она поднята.
    """
    choice = os.environ.get("ANALYSIS_PROVIDER", "").strip().lower()
    if choice == "demo":
        return "demo"
    if choice == "claude":
        return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "demo"
    if choice == "local":
        from meeting import analyze_local
        return "local" if analyze_local.available() else "demo"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    from meeting import analyze_local
    return "local" if analyze_local.available() else "demo"


def mode():
    """Что сейчас доступно — показываем честно и в интерфейсе, и в логах."""
    return {
        "transcription": stt_provider(),
        "analysis": analysis_provider(),
    }


# --------------------------------------------------------------------------
# Распознавание речи
# --------------------------------------------------------------------------

def _soniox_request(path, payload=None, method="POST", api_key=""):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(SONIOX_BASE + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + api_key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def _soniox_upload(audio_path, api_key):
    """Загрузка файла multipart/form-data силами стандартной библиотеки."""
    boundary = "----hackalem" + uuid.uuid4().hex
    audio_path = Path(audio_path)
    head = (
        "--{b}\r\n"
        'Content-Disposition: form-data; name="file"; filename="{n}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).format(b=boundary, n=audio_path.name).encode("utf-8")
    tail = "\r\n--{b}--\r\n".format(b=boundary).encode("utf-8")
    body = head + audio_path.read_bytes() + tail

    req = urllib.request.Request(SONIOX_BASE + "/v1/files", data=body, method="POST")
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))["id"]


def _tokens_to_segments(tokens):
    """Склеиваем токены в реплики: новая реплика — когда сменился говорящий."""
    segments = []
    for tok in tokens:
        # Soniox отдаёт безличные метки «1», «2» — сразу делаем их читаемыми.
        raw = str(tok.get("speaker") or "").strip()
        speaker = "Спикер " + raw if raw.isdigit() else (raw or "Спикер")
        text = tok.get("text", "")
        start = (tok.get("start_ms") or 0) / 1000.0
        end = (tok.get("end_ms") or 0) / 1000.0
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["text"] += text
            segments[-1]["end"] = end
        else:
            segments.append({"start": start, "end": end, "speaker": speaker, "text": text})
    for seg in segments:
        seg["text"] = re.sub(r"\s+", " ", seg["text"]).strip()
    return [s for s in segments if s["text"]]


def transcribe(audio_path=None, title=None):
    """Возвращает {title, duration_sec, segments, transcription_mode}."""
    provider = stt_provider()

    if not audio_path:  # кнопка «Разобрать пример»
        sample = json.loads((SAMPLES / "planerka.json").read_text(encoding="utf-8"))
        sample["transcription_mode"] = "demo"
        if title:
            sample["title"] = title
        return sample

    if provider == "demo":
        # Человек загрузил свою запись — молча подсунуть вместо неё демо-пример
        # хуже, чем честно сказать, чего не хватает.
        raise RuntimeError(
            "Загруженную запись распознать нечем: локальные модели не установлены. "
            "Выполните: python scripts/get_models.py — либо укажите STT_PROVIDER=soniox "
            "и ключ SONIOX_API_KEY. Кнопка «Разобрать пример» работает без моделей.")

    if provider == "local":
        from meeting import stt_local
        return stt_local.transcribe(audio_path, title)

    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    file_id = _soniox_upload(audio_path, api_key)
    started = _soniox_request("/v1/transcriptions", {
        "file_id": file_id,
        "model": os.environ.get("SONIOX_MODEL", "stt-async-v5"),
        "language_hints": [
            h.strip() for h in os.environ.get("SONIOX_LANGUAGE_HINTS", "ru,kk").split(",") if h.strip()
        ],
        "enable_speaker_diarization": True,
    }, api_key=api_key)

    transcription_id = started["id"]
    for _ in range(POLL_ATTEMPTS):
        status = _soniox_request("/v1/transcriptions/" + transcription_id,
                                 method="GET", api_key=api_key)
        state = status.get("status")
        if state == "completed":
            break
        if state == "error":
            raise RuntimeError("Soniox: " + str(status.get("error_message")))
        time.sleep(POLL_SECONDS)
    else:
        raise RuntimeError("Soniox: не дождались расшифровки")

    result = _soniox_request("/v1/transcriptions/" + transcription_id + "/transcript",
                             method="GET", api_key=api_key)
    segments = _tokens_to_segments(result.get("tokens", []))
    return {
        "title": title or Path(audio_path).stem,
        "duration_sec": round(segments[-1]["end"], 1) if segments else 0,
        "segments": segments,
        "transcription_mode": "soniox",
    }


# --------------------------------------------------------------------------
# Извлечение протокола и поручений
# --------------------------------------------------------------------------

PROMPT = """Ты составляешь протокол рабочего совещания на русском языке.

Тебе дана стенограмма с таймкодами и говорящими. Верни СТРОГО JSON:

{
  "speakers": {"метка говорящего": "имя или null"},
  "title": "короткое название совещания",
  "summary": "2-3 предложения: о чём договорились",
  "decisions": ["принятое решение", ...],
  "action_items": [
    {
      "what": "что именно нужно сделать, одной фразой",
      "who": "имя ответственного или null, если не назначен",
      "due": "срок словами как в разговоре или null",
      "priority": "high|medium|low",
      "quote": "ТОЧНАЯ цитата из стенограммы, откуда это следует",
      "confidence": 0.0-1.0
    }
  ]
}

Жёсткие правила:
1. В "quote" — дословный фрагмент из стенограммы. Не перефразируй: цитата проверяется автоматически.
2. Не выдумывай ответственных и сроки. Не названо — ставь null и confidence ниже 0.6.
3. Поручение — это то, что кто-то обязался или кого-то попросили сделать. Общие рассуждения не поручения.
4. Если поручений нет, верни пустой список.
5. "speakers": расшифровка говорящих. Распознавание речи даёт безличные метки
   («1», «2»), а имена звучат в разговоре при обращении: если кто-то говорит
   «Тимур, сможешь…», то отвечающий следующим — Тимур. Имя, которое в
   стенограмме не прозвучало, не придумывай: ставь null.
6. Если человек берёт задачу на себя («я сам этим займусь», «я подготовлю»),
   ответственный — тот, кто это сказал. Подставь его имя, а если оно неизвестно —
   метку его говорящего.

Стенограмма:
{transcript}
"""

# Обороты, которыми человек берёт задачу на себя: ответственный — сам говорящий.
SELF_ASSIGN = ("я сам", "я сама", "я займ", "я возьм", "беру на себя",
               "я подготов", "я сделаю", "я договор", "я напишу", "я посмотрю")


def _format_transcript(segments):
    return "\n".join(
        "[{:.1f}] {}: {}".format(s["start"], s["speaker"], s["text"]) for s in segments
    )


def _normalize(text):
    return re.sub(r"[^a-zа-я0-9 ]+", " ", (text or "").lower().replace("ё", "е"))


def _ground(items, segments):
    """Проверяем каждую цитату по стенограмме и сами проставляем таймкод.

    Модель может ошибиться или придумать — поэтому таймкоду из её ответа мы не
    верим, а ищем цитату в реальном тексте. Не нашли — помечаем как непроверенное.
    """
    normalized = [(_normalize(s["text"]), s) for s in segments]
    for item in items:
        quote = _normalize(item.get("quote", ""))
        item["grounded"] = False
        if len(quote.split()) >= 3:
            for norm_text, seg in normalized:
                if quote in norm_text:
                    item["grounded"] = True
                    item["start"] = seg["start"]
                    item["said_by"] = seg["speaker"]
                    break
        if not item["grounded"]:
            item["confidence"] = min(float(item.get("confidence") or 0.5), 0.4)
            item.setdefault("start", 0)
    return items


def _apply_speaker_names(speakers, segments):
    """Подставляем распознанные имена вместо безличных меток Soniox.

    Имя принимаем только если оно действительно звучало в стенограмме — тот же
    принцип, что и с цитатами: модель предлагает, код проверяет.
    """
    if not isinstance(speakers, dict):
        return {}
    spoken = _normalize(" ".join(s["text"] for s in segments)).split()
    applied = {}
    for label, name in speakers.items():
        name = (name or "").strip()
        key = _normalize(name).strip()
        # startswith — чтобы «Тимуру» в тексте подтвердило имя «Тимур»
        if key and any(word.startswith(key) for word in spoken):
            applied[str(label)] = name
    if applied:
        for seg in segments:
            seg["speaker"] = applied.get(str(seg["speaker"]), seg["speaker"])
    return applied


def _fill_self_assigned(items):
    """«Я сам этим займусь» — ответственный тот, кто это произнёс."""
    for item in items:
        if not item.get("who") and item.get("said_by"):
            quote = _normalize(item.get("quote", ""))
            if any(marker in quote for marker in SELF_ASSIGN):
                item["who"] = item["said_by"]
                item["who_from_speaker"] = True
    return items


def _parse_json(raw):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("модель вернула не JSON: " + raw[:200])
    return json.loads(match.group(0))


def extract(transcript):
    """Возвращает {title, summary, decisions, action_items, analysis_mode}."""
    segments = transcript["segments"]
    provider = analysis_provider()
    prompt = PROMPT.replace("{transcript}", _format_transcript(segments))

    if provider == "demo":
        result = json.loads((SAMPLES / "planerka.analysis.json").read_text(encoding="utf-8"))
        result["analysis_mode"] = "demo"
        result["action_items"] = _ground(result["action_items"], segments)
        return result

    if provider == "local":
        from meeting import analyze_local
        result = analyze_local.analyze(prompt)
        result["analysis_mode"] = "local"
    else:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Не установлен пакет anthropic: pip install -r requirements.txt")

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
        response = client.messages.create(
            model=os.environ.get("ANALYSIS_MODEL", DEFAULT_MODEL),
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        result = _parse_json(text)
        result["analysis_mode"] = "claude"
    result["speakers"] = _apply_speaker_names(result.get("speakers"), segments)
    result["action_items"] = _fill_self_assigned(
        _ground(result.get("action_items", []), segments))
    return result


def process(audio_path=None, title=None):
    """Полный проход: аудио (или демо-пример) -> готовая карточка совещания."""
    transcript = transcribe(audio_path, title)
    analysis = extract(transcript)

    items = []
    for index, item in enumerate(analysis.get("action_items", [])):
        item["id"] = "task-{}".format(index + 1)
        item["status"] = "pending"
        items.append(item)

    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": analysis.get("title") or transcript.get("title") or "Совещание",
        "summary": analysis.get("summary", ""),
        "decisions": analysis.get("decisions", []),
        "action_items": items,
        "segments": transcript["segments"],
        "duration_sec": transcript.get("duration_sec", 0),
        "modes": {
            "transcription": transcript.get("transcription_mode", "demo"),
            "analysis": analysis.get("analysis_mode", "demo"),
        },
    }
