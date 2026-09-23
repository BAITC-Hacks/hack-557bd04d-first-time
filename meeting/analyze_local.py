"""Локальный разбор совещания через Ollama — без облака.

Закрывает последнюю часть ограничения ТЗ: распознавание речи уже идёт на машине,
а здесь на машине остаётся и разбор текста. Модель работает по тому же контракту,
что и облачный провайдер, поэтому остальной код о подмене не знает.

Ollama слушает localhost:11434 и наружу ничего не отправляет.

    ollama serve
    ollama pull qwen2.5:7b-instruct
    ANALYSIS_PROVIDER=local python app.py
"""

import json
import os
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("LOCAL_ANALYSIS_MODEL", "qwen2.5:7b-instruct")
TIMEOUT = int(os.environ.get("LOCAL_ANALYSIS_TIMEOUT", "900"))


class LocalAnalysisError(RuntimeError):
    pass


def _request(path, payload=None, method="POST"):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def available():
    """Поднят ли Ollama и есть ли нужная модель."""
    try:
        tags = _request("/api/tags", method="GET")
    except Exception:
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    # ollama допускает и короткое имя без тега
    return bool(names) and (MODEL in names or any(n.split(":")[0] == MODEL.split(":")[0]
                                                  for n in names))


def models():
    try:
        return sorted(m.get("name", "") for m in _request("/api/tags", method="GET").get("models", []))
    except Exception:
        return []


def analyze(prompt):
    """Отдаёт разбор словарём. Формат ответа тот же, что у облачного провайдера."""
    try:
        response = _request("/api/generate", {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            # Ollama умеет принудительно выдавать валидный JSON — это снимает
            # половину проблем маленьких моделей с форматом ответа.
            "format": "json",
            "options": {
                "temperature": 0,
                "num_ctx": 8192,   # стенограмма часового совещания сюда влезает
                "num_predict": 2048,
            },
        })
    except urllib.error.URLError as exc:
        raise LocalAnalysisError(
            "Ollama недоступен на {}: {}. Запустите `ollama serve`".format(HOST, exc.reason))

    raw = (response.get("response") or "").strip()
    if not raw:
        raise LocalAnalysisError("модель {} вернула пустой ответ".format(MODEL))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalAnalysisError("модель вернула не JSON: {} ({})".format(raw[:200], exc))
