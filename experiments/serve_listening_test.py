"""
experiments/serve_listening_test.py
===================================
Serve the listening test on this machine and collect submissions directly.

WHY A SERVER AND NOT JUST THE HTML FILE
---------------------------------------
Two reasons, both practical rather than architectural.

1. Opening `index.html` over `file://` works in some browsers and silently
   fails to load the audio in others, depending on local-file policy. Over HTTP
   it behaves the same everywhere.
2. Without somewhere to POST to, every listener has to download a CSV and send
   it back, and someone has to file it in `responses/` under the right name.
   That is the step that loses data in a small unpaid panel. With this running,
   a submission lands in `listening_test/responses/` by itself, and the page
   still falls back to the download and copy buttons if the POST fails.

SCOPE, DELIBERATELY SMALL
-------------------------
* Binds **127.0.0.1 by default.** Colleagues on other machines need
  `--host 0.0.0.0`, which is an explicit choice to expose the test on the local
  network, not something this script does for you. It serves the
  `listening_test/` directory and nothing above it.
* There is **no authentication**, so do not leave it running on an untrusted
  network, and do not put anything in `listening_test/` that should not be
  readable by anyone who can reach the port.
* POST bodies are capped, the listener id is sanitised before it reaches a
  filename, and an existing file is never overwritten -- a repeat submission
  gets a new suffix, and the analysis script reports duplicate ids rather than
  averaging them.

Run:
  python experiments/serve_listening_test.py
  python experiments/serve_listening_test.py --host 0.0.0.0 --port 8123
Then send colleagues the URL it prints. Stop it with Ctrl-C.
Analyse with:
  python experiments/analyse_listening.py
"""
import argparse
import datetime as dt
import os
import re
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT = os.path.join(PROJ, "listening_test")
RESP = os.path.join(LT, "responses")
MAX_BODY = 2 * 1024 * 1024          # a full submission is ~8 KB
SAFE = re.compile(r"[^A-Za-z0-9_.-]")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") not in ("/submit",):
            self.send_error(404, "no such endpoint")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self.send_error(413, "missing or oversized body")
            return
        body = self.rfile.read(n).decode("utf-8", errors="replace")

        # The listener id comes from the first data row, so the filename says
        # who it is without trusting a header or a query string.
        lines = [l for l in body.splitlines() if l.strip()]
        who = "anon"
        if len(lines) > 1:
            who = (lines[1].split(",") or ["anon"])[0] or "anon"
        who = SAFE.sub("_", who)[:40] or "anon"

        os.makedirs(RESP, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESP, f"listening_{who}_{stamp}.csv")
        k = 1
        while os.path.exists(path):      # never clobber a submission
            path = os.path.join(RESP, f"listening_{who}_{stamp}_{k}.csv")
            k += 1
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)

        print(f"  [submission] {who}: {len(lines) - 1} ratings -> "
              f"{os.path.basename(path)}", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):      # one line per submission is enough
        pass


def lan_ips():
    out = []
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in out and not ip.startswith("127."):
                out.append(ip)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) keeps the test on this machine; "
                         "0.0.0.0 exposes it to the local network")
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args()

    page = os.path.join(LT, "index.html")
    if not os.path.isfile(page):
        raise SystemExit(f"[serve] {page} missing; run "
                         f"experiments/make_listening_page.py first")
    n_stim = len([f for f in os.listdir(os.path.join(LT, "stimuli"))
                  if f.endswith(".wav")]) if os.path.isdir(
        os.path.join(LT, "stimuli")) else 0
    if not n_stim:
        raise SystemExit("[serve] listening_test/stimuli/ is empty; run "
                         "experiments/make_listening_stimuli.py first")

    os.makedirs(RESP, exist_ok=True)
    have = len([f for f in os.listdir(RESP) if f.endswith(".csv")])
    handler = partial(Handler, directory=LT)
    srv = ThreadingHTTPServer((a.host, a.port), handler)

    print(f"[serve] listening test on http://{a.host}:{a.port}/")
    print(f"[serve] {n_stim} stimuli | {have} submission(s) already collected")
    if a.host == "0.0.0.0":
        for ip in lan_ips():
            print(f"[serve]   reachable from this network at "
                  f"http://{ip}:{a.port}/")
        print("[serve] NOTE: no authentication. Anyone who can reach this port "
              "can take the test and read listening_test/.")
    else:
        print("[serve] bound to localhost only. Pass --host 0.0.0.0 to let "
              "other machines reach it.")
    print(f"[serve] submissions are written to {RESP}")
    print("[serve] Ctrl-C to stop; then: python experiments/analyse_listening.py")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
