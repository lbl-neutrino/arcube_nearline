#!/usr/bin/env python3

"""Minimal mTLS PUT upload service (Python standard library only).

Authentication is done entirely by the TLS handshake: the trust anchor is a
single end-entity user cert (EEC) in OpenSSL "TRUSTED CERTIFICATE" form, and
proxy certs are permitted, so ONLY proxies delegated from that one EEC verify.

Anchor must be the trusted-cert form of the EEC, e.g.:
    openssl x509 -in eec.crt -addtrust anyExtendedKeyUsage -out eec_trusted.pem

For justIN jobs, the EEC is justIN's annually-renewed key used for signing X509
proxies. This can be obtained by taking the top-level certificate from the result
of "justin get-token".

There is no application-level authorization -- if a request reaches the handler,
the client already presented a proxy chaining to the pinned EEC.

There is one route:

    PUT /path  ->  stores the request body at UPLOAD_DIR/path
"""

import os
import ssl
import secrets
import posixpath
from urllib.parse import urlsplit, unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = "0.0.0.0", 55556
SERVER_CERT, SERVER_KEY = "server.crt", "server.key"
TRUST_ANCHORS = "justin_eec_trusted.pem"
UPLOAD_DIR = "uploads"
MAX_UPLOAD = 10 * 1024 * 1024


def safe_upload_path(url_path):
    """Map a request URI to a path strictly inside UPLOAD_DIR, or None if unsafe."""
    raw = unquote(urlsplit(url_path).path)
    if raw.endswith("/"):                     # can't PUT a directory
        return None
    norm = posixpath.normpath(raw).lstrip("/")
    if not norm or norm.startswith("..") or "/../" in norm:
        return None
    real_upload_dir = os.path.realpath(UPLOAD_DIR)
    dest = os.path.realpath(os.path.join(real_upload_dir, norm))
    if os.path.commonpath([real_upload_dir, dest]) != real_upload_dir:
        return None
    return dest


class Handler(BaseHTTPRequestHandler):

    def _reply(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _write(self, dest, length):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.part"
        remaining = length
        try:
            with open(tmp, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            if remaining != 0:
                os.unlink(tmp)
                self._reply(400, b"Incomplete body\n")
                raise
            os.replace(tmp, dest)             # atomic
        except Exception as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.log_message("upload error: %r", e)
            self._reply(500, b"Upload failed\n")
            raise

    def do_PUT(self):
        dest = safe_upload_path(self.path)
        if dest is None:
            self._reply(400, b"Invalid upload path\n")
            return
        existed = os.path.exists(dest)

        try:
            length = int(self.headers["Content-Length"])
        except (KeyError, TypeError, ValueError):
            self._reply(411, b"Content-Length required\n")
            return
        if length > MAX_UPLOAD:
            self._reply(413, b"Payload too large\n")
            return

        try:
            self._write(dest, length)
        except:
            return

        self.log_message("PUT %s (%d bytes)", dest, length)
        self._reply(204 if existed else 201, b"" if existed else b"Created\n")

    def log_message(self, format, *args):
        print(format % args)


def make_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(SERVER_CERT, SERVER_KEY)
    ctx.load_verify_locations(TRUST_ANCHORS)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags |= ssl.VERIFY_ALLOW_PROXY_CERTS # crucial!!!
    return ctx


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.socket = make_context().wrap_socket(httpd.socket, server_side=True)
    print(f"Listening on https://{HOST}:{PORT}  (uploads -> {UPLOAD_DIR})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
