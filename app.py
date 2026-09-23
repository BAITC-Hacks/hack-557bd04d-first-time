#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Протокол совещаний: веб-сервер на стандартной библиотеке.

Запуск:  python app.py       далее http://localhost:8000
"""

import base64
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meeting import export_docx, pipeline, store  # noqa: E402

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY = 200 * 1024 * 1024
DOCX_TYPE = ("application/vnd.openxmlformats-officedocument"
             ".wordprocessingml.document")


class Handler(BaseHTTPRequestHandler):
    server_version = "MeetingMinutes/0.1"

    # ---------- вспомогательное ----------

    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        """Битое тело не должно ронять обработчик — отвечаем пустым объектом."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    # ---------- маршруты ----------

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_text(encoding="utf-8"),
                              "text/html; charset=utf-8")

        if path == "/api/health":
            return self._send(200, {"status": "ok", "modes": pipeline.mode()})

        if path == "/api/meetings":
            return self._send(200, store.list_all())

        if path.endswith("/export.docx"):
            meeting = store.load(path.split("/")[3])
            if not meeting:
                return self._send(404, {"error": "не найдено"})
            data = export_docx.build(meeting)
            name = "protokol-%s.docx" % meeting["id"]
            self.send_response(200)
            self.send_header("Content-Type", DOCX_TYPE)
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        if path.startswith("/api/meetings/"):
            meeting = store.load(path.rsplit("/", 1)[-1])
            return self._send(200, meeting) if meeting else self._send(404, {"error": "не найдено"})

        return self._send(404, {"error": "не найдено"})

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/analyze":
            return self._analyze()

        parts = [p for p in path.split("/") if p]

        # /api/meetings/<id>/speakers — дать говорящему имя вручную
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "meetings" and parts[3] == "speakers":
            payload = self._read_json()
            old, new = payload.get("from", ""), payload.get("to", "")
            if not old or not new.strip():
                return self._send(400, {"error": "нужны непустые поля from и to"})
            meeting = store.rename_speaker(parts[2], old, new)
            return self._send(200, meeting) if meeting else self._send(404, {"error": "не найдено"})

        # /api/meetings/<id>/items/<item_id>
        if len(parts) == 5 and parts[0] == "api" and parts[1] == "meetings" and parts[3] == "items":
            status = (self._read_json().get("status") or "").strip()
            if status not in ("pending", "confirmed", "rejected"):
                return self._send(400, {"error": "статус должен быть pending, confirmed или rejected"})
            meeting = store.set_status(parts[2], parts[4], status)
            return self._send(200, meeting) if meeting else self._send(404, {"error": "не найдено"})

        return self._send(404, {"error": "не найдено"})

    def _analyze(self):
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "ожидался JSON"})

        audio_path = None
        tmp = None
        try:
            if payload.get("audio_b64"):
                suffix = Path(payload.get("filename") or "audio.wav").suffix or ".wav"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(base64.b64decode(payload["audio_b64"]))
                tmp.close()
                audio_path = tmp.name

            meeting = pipeline.process(audio_path, payload.get("title"))
            store.save(meeting)
            return self._send(200, meeting)
        except Exception as exc:  # показываем причину, а не пустой экран
            self.log_message("ошибка разбора: %s", exc)
            return self._send(500, {"error": str(exc)})
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass


def main():
    pipeline.load_env()
    modes = pipeline.mode()
    print("Протокол совещаний  ->  http://localhost:{}".format(PORT))
    print("  распознавание речи: {}".format(modes["transcription"]))
    print("  разбор совещания:   {}".format(modes["analysis"]))
    if "demo" in modes.values():
        print("  (демо-режим: работает без ключей на примере из samples/)")

    # Греем модели в фоне: сервер отвечает сразу, а к первому разбору они
    # уже в памяти. Без этого первый пользователь ждёт загрузку с диска.
    if modes["transcription"] == "local":
        import threading
        from meeting import stt_local
        print("  прогреваю модели распознавания в фоне…")
        threading.Thread(target=stt_local.warmup, daemon=True).start()

    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
