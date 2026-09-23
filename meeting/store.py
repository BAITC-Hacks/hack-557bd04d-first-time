"""Хранилище совещаний: по одному JSON-файлу на совещание.

База данных для прототипа была бы лишней зависимостью при установке у проверяющего.
"""

import json
from pathlib import Path

from meeting import deadlines

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage" / "meetings"


def save(meeting):
    deadlines.annotate(meeting)
    STORAGE.mkdir(parents=True, exist_ok=True)
    path = STORAGE / (meeting["id"] + ".json")
    path.write_text(json.dumps(meeting, ensure_ascii=False, indent=2), encoding="utf-8")
    return meeting


def load(meeting_id):
    path = STORAGE / (meeting_id + ".json")
    if not path.exists():
        return None
    # Статусы сроков пересчитываем при каждом чтении: вчерашнее «осталось 2 дня»
    # сегодня уже неправда.
    return deadlines.annotate(json.loads(path.read_text(encoding="utf-8")))


def list_all():
    if not STORAGE.exists():
        return []
    meetings = []
    for path in STORAGE.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        meetings.append({
            "id": data["id"],
            "title": data["title"],
            "created_at": data.get("created_at", ""),
            "tasks": len(data.get("action_items", [])),
        })
    return sorted(meetings, key=lambda m: m["created_at"], reverse=True)


def rename_speaker(meeting_id, old, new):
    """Дать говорящему имя вручную.

    Распознавание речи различает голоса, но имя знает только если оно прозвучало
    вслух. Когда не прозвучало — имя ставит человек, и оно сразу расходится по
    стенограмме и по ответственным за поручения.
    """
    meeting = load(meeting_id)
    if not meeting or not new.strip():
        return None
    new = new.strip()
    for seg in meeting.get("segments", []):
        if seg.get("speaker") == old:
            seg["speaker"] = new
    for item in meeting.get("action_items", []):
        for field in ("who", "said_by"):
            if item.get(field) == old:
                item[field] = new
    speakers = meeting.get("speakers") or {}
    meeting["speakers"] = {k: (new if v == old else v) for k, v in speakers.items()}
    return save(meeting)


def set_status(meeting_id, item_id, status):
    """Человек подтверждает или отклоняет поручение — решение остаётся за ним."""
    meeting = load(meeting_id)
    if not meeting:
        return None
    for item in meeting.get("action_items", []):
        if item.get("id") == item_id:
            item["status"] = status
            save(meeting)
            return meeting
    return None
