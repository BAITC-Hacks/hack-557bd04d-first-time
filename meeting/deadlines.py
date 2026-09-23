"""Сроки поручений: из слов в дату, из даты в статус контроля.

Второй сценарий задания — руководитель должен получать напоминание при
приближении срока или при просрочке. Для этого срок, сказанный в разговоре
словами («до пятницы», «к пятнадцатому октября»), нужно превратить в дату.

Считаем от даты совещания: «до пятницы» — это ближайшая пятница после него,
а не после сегодняшнего дня. Статус же считаем от сегодня, потому что он
отвечает на вопрос «что горит прямо сейчас».

Чего мы здесь намеренно не делаем — не угадываем. Не разобрали формулировку,
значит срока нет: лучше пустое поле, чем правдоподобная выдумка в отчёте.
"""

import datetime
import re

WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среда": 2, "среду": 2, "четверг": 3,
    "пятница": 4, "пятницу": 4, "суббота": 5, "субботу": 5,
    "воскресенье": 6, "понедельника": 0, "вторника": 1, "среды": 2,
    "четверга": 3, "пятницы": 4, "субботы": 5, "воскресенья": 6,
}

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12, "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
    "май": 5, "июнь": 6, "июль": 7, "август": 8, "сентябрь": 9,
    "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}

# Порядковые числительные словами — именно так сроки и звучат в речи.
# Сравниваем по основе: «пятнадцатого», «пятнадцатому», «пятнадцатое» — одно и то же.
# Длинные основы проверяем первыми, иначе «пят» перехватит «пятнадцат».
ORDINAL_STEMS = sorted([
    ("перв", 1), ("втор", 2), ("трет", 3), ("четверт", 4),
    ("пятнадцат", 15), ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18),
    ("девятнадцат", 19), ("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13),
    ("четырнадцат", 14), ("двадцат", 20), ("двадцать", 20), ("тридцат", 30),
    ("тридцать", 30), ("пят", 5), ("шест", 6), ("седьм", 7), ("восьм", 8),
    ("девят", 9), ("десят", 10),
], key=lambda pair: -len(pair[0]))


def _ordinal(word):
    """День месяца по основе слова, иначе None."""
    for stem, value in ORDINAL_STEMS:
        if word.startswith(stem):
            return value
    return None

SOON_DAYS = 2  # за сколько дней до срока поручение считается горящим


def _normalize(text):
    text = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^а-яa-z0-9 ]+", " ", text)).strip()


def _next_weekday(base, weekday):
    """Ближайший такой день недели строго после даты совещания."""
    ahead = (weekday - base.weekday()) % 7
    return base + datetime.timedelta(days=ahead or 7)


def parse(due_text, meeting_date):
    """Слова -> дата. Не поняли формулировку — возвращаем None, не выдумываем."""
    text = _normalize(due_text)
    if not text or text in ("не назван", "не указано", "нет"):
        return None

    if "послезавтра" in text:
        return meeting_date + datetime.timedelta(days=2)
    if "завтра" in text:
        return meeting_date + datetime.timedelta(days=1)
    if "сегодня" in text:
        return meeting_date

    # «до 15 октября», «к 3 ноября»
    match = re.search(r"(\d{1,2})\s+([а-я]+)", text)
    if match and match.group(2) in MONTHS:
        return _build(meeting_date, int(match.group(1)), MONTHS[match.group(2)])

    # «до пятнадцатого октября» — числительное словами
    words = text.split()
    for index, word in enumerate(words):
        day = _ordinal(word)
        if day is None:
            continue
        # «двадцать шестого» — составное числительное
        if day in (20, 30) and index + 1 < len(words):
            second = _ordinal(words[index + 1])
            if second and second < 10:
                day += second
        month = next((MONTHS[w] for w in words[index:] if w in MONTHS), None)
        if month:
            return _build(meeting_date, day, month)

    for name, weekday in WEEKDAYS.items():
        if name in text:
            base = meeting_date
            if "следующ" in text:
                base = meeting_date + datetime.timedelta(days=7)
            return _next_weekday(base, weekday)

    if "конца недели" in text:
        return _next_weekday(meeting_date, 4)
    if "следующей недел" in text:
        return meeting_date + datetime.timedelta(days=7)
    if "конца месяца" in text:
        following = (meeting_date.replace(day=28) + datetime.timedelta(days=4))
        return following - datetime.timedelta(days=following.day)

    match = re.search(r"(\d+)\s*(дн|недел|мес)", text)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        days = count if unit == "дн" else count * 7 if unit == "недел" else count * 30
        return meeting_date + datetime.timedelta(days=days)

    return None


def _build(meeting_date, day, month):
    """Год в разговоре не называют — берём ближайший подходящий."""
    year = meeting_date.year
    try:
        result = datetime.date(year, month, day)
    except ValueError:
        return None
    if (result - meeting_date).days < -180:  # «в январе», сказанное в декабре
        result = datetime.date(year + 1, month, day)
    return result


def status(due_date, today=None):
    """Что показывать руководителю: просрочено, горит, в работе."""
    if due_date is None:
        return {"code": "unknown", "label": "срок не назван", "days_left": None}
    today = today or datetime.date.today()
    days = (due_date - today).days
    if days < 0:
        return {"code": "overdue", "label": "просрочено на %d дн." % abs(days), "days_left": days}
    if days == 0:
        return {"code": "today", "label": "срок сегодня", "days_left": 0}
    if days <= SOON_DAYS:
        return {"code": "soon", "label": "осталось %d дн." % days, "days_left": days}
    return {"code": "planned", "label": "осталось %d дн." % days, "days_left": days}


def annotate(meeting, today=None):
    """Проставляет каждому поручению дату срока и статус контроля."""
    raw = (meeting.get("created_at") or "")[:10]
    try:
        meeting_date = datetime.datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        meeting_date = datetime.date.today()

    counters = {"overdue": 0, "today": 0, "soon": 0, "planned": 0, "unknown": 0}
    for item in meeting.get("action_items", []):
        due_date = parse(item.get("due"), meeting_date)
        # Отклонённое поручение никого не торопит.
        state = status(due_date, today) if item.get("status") != "rejected" else {
            "code": "planned", "label": "отклонено", "days_left": None}
        item["due_date"] = due_date.isoformat() if due_date else None
        item["deadline"] = state
        counters[state["code"]] = counters.get(state["code"], 0) + 1
    meeting["deadline_summary"] = counters
    return meeting
