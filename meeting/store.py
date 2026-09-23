"""Хранилище совещаний: по одному JSON-файлу на совещание.

База данных для прототипа была бы лишней зависимостью при установке у проверяющего.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage" / "meetings"


def save(meeting):
    STORAGE.mkdir(parents=True, exist_ok=True)
    path = STORAGE / (meeting["id"] + ".json")
    path.write_text(json.dumps(meeting, ensure_ascii=False, indent=2), encoding="utf-8")
    return meeting


def load(meeting_id):
    path = STORAGE / (meeting_id + ".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
