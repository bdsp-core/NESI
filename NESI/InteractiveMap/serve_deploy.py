"""
Tiny basic-auth static-file server for the deploy/ folder. Designed to
sit behind an ngrok tunnel so colleagues can view the explorer via a
public URL gated by user/pass.

  Usage:
    cd NESI/InteractiveMap
    YAMA_USER=brandon YAMA_PASS=somepass python3 serve_deploy.py
    # in another shell:
    ngrok http 8000
"""
from __future__ import annotations

import base64
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"

USER = os.environ.get("YAMA_USER", "yama")
PASS = os.environ.get("YAMA_PASS")
PORT = int(os.environ.get("YAMA_PORT", "8000"))

if not PASS:
    sys.exit("Set YAMA_PASS env var before starting the server.")


class AuthHandler(SimpleHTTPRequestHandler):
    REALM = "YAMA Explorer"

    def _require_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                user, pw = (base64.b64decode(auth[6:]).decode()
                              .split(":", 1))
                if user == USER and pw == PASS:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate",
                          f'Basic realm="{self.REALM}"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self):
        if not self._require_auth():
            return
        super().do_GET()

    def do_HEAD(self):
        if not self._require_auth():
            return
        super().do_HEAD()

    def log_message(self, fmt, *args):
        # Quiet the noisy per-request logging.
        if any(s in fmt % args for s in ('200', '304')):
            return
        super().log_message(fmt, *args)


if __name__ == "__main__":
    if not DEPLOY_DIR.exists():
        sys.exit(f"Missing deploy directory: {DEPLOY_DIR}")
    os.chdir(DEPLOY_DIR)
    print(f"Serving {DEPLOY_DIR} on http://localhost:{PORT}/")
    print(f"Basic auth: user={USER}  pass={'*' * len(PASS)}")
    HTTPServer(("", PORT), AuthHandler).serve_forever()
