"""Экспорт протокола в DOCX — обязательный пункт задания.

Формат .docx — это zip с несколькими XML-файлами, поэтому собираем его
стандартной библиотекой. Ставить Word или тянуть внешний пакет не нужно,
и проверяющему тоже не придётся.
"""

import io
import zipfile
from xml.sax.saxutils import escape

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _p(text, bold=False, size=22, space_after=120):
    """Абзац. size — половины пункта: 22 это 11pt."""
    runs = '<w:rPr>{}<w:sz w:val="{}"/></w:rPr>'.format("<w:b/>" if bold else "", size)
    return (
        '<w:p><w:pPr><w:spacing w:after="{after}"/></w:pPr>'
        '<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
    ).format(after=space_after, rpr=runs, text=escape(str(text or "")))


def _bullet(text):
    return (
        '<w:p><w:pPr><w:ind w:left="360"/><w:spacing w:after="60"/></w:pPr>'
        '<w:r><w:rPr><w:sz w:val="22"/></w:rPr>'
        '<w:t xml:space="preserve">• {}</w:t></w:r></w:p>'
    ).format(escape(str(text or "")))


def _cell(text, bold=False, width=2000):
    return (
        '<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>'
        '<w:p><w:pPr><w:spacing w:after="0"/></w:pPr><w:r>'
        '<w:rPr>{b}<w:sz w:val="20"/></w:rPr>'
        '<w:t xml:space="preserve">{t}</w:t></w:r></w:p></w:tc>'
    ).format(w=width, b="<w:b/>" if bold else "", t=escape(str(text or "")))


def _table(rows, widths):
    borders = "".join(
        '<w:{} w:val="single" w:sz="4" w:color="BFBFBF"/>'.format(side)
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
    # tblGrid обязателен по схеме OOXML — без него строгие парсеры файл отвергают.
    grid = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
    body = ""
    for index, row in enumerate(rows):
        cells = "".join(_cell(v, bold=(index == 0), width=w) for v, w in zip(row, widths))
        body += "<w:tr>%s</w:tr>" % cells
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>{borders}</w:tblBorders>'
        '</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>'
    ).format(borders=borders, grid=grid, body=body)


STATUS_LABEL = {"confirmed": "подтверждено", "rejected": "отклонено", "pending": "на проверке"}


def _fmt_time(seconds):
    total = int(seconds or 0)
    return "%02d:%02d" % (total // 60, total % 60)


def build(meeting, include_transcript=True):
    """Собирает .docx и возвращает его байтами."""
    parts = [_p(meeting.get("title") or "Протокол совещания", bold=True, size=32, space_after=200)]

    meta = "Дата разбора: {} · длительность {} · поручений: {}".format(
        meeting.get("created_at", ""), _fmt_time(meeting.get("duration_sec")),
        len(meeting.get("action_items", [])))
    parts.append(_p(meta, size=18, space_after=240))

    if meeting.get("summary"):
        parts.append(_p("Кратко", bold=True, size=26))
        parts.append(_p(meeting["summary"]))

    decisions = meeting.get("decisions") or []
    if decisions:
        parts.append(_p("Решения", bold=True, size=26, space_after=80))
        parts.extend(_bullet(d) for d in decisions)
        parts.append(_p("", space_after=120))

    items = meeting.get("action_items") or []
    parts.append(_p("Поручения", bold=True, size=26, space_after=120))
    if items:
        rows = [["Что", "Ответственный", "Срок", "Таймкод", "Статус"]]
        for item in items:
            rows.append([
                item.get("what", ""),
                item.get("who") or "не назначен",
                item.get("due") or "не назван",
                _fmt_time(item.get("start")),
                STATUS_LABEL.get(item.get("status"), item.get("status", "")),
            ])
        parts.append(_table(rows, [4200, 1900, 1500, 900, 1400]))
        parts.append(_p("", space_after=160))
        # Цитаты — чтобы поручение можно было проверить по записи, а не верить на слово.
        parts.append(_p("Основания", bold=True, size=24, space_after=80))
        for item in items:
            parts.append(_bullet("[{}] {} — «{}»".format(
                _fmt_time(item.get("start")), item.get("what", ""), item.get("quote", ""))))
    else:
        parts.append(_p("Поручений не зафиксировано."))

    if include_transcript and meeting.get("segments"):
        parts.append(_p("Стенограмма", bold=True, size=26, space_after=120))
        for seg in meeting["segments"]:
            parts.append(_p("[{}] {}: {}".format(
                _fmt_time(seg.get("start")), seg.get("speaker", ""), seg.get("text", "")),
                size=20, space_after=60))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document {w}><w:body>{body}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        '</w:body></w:document>'
    ).format(w=W, body="".join(parts))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()
