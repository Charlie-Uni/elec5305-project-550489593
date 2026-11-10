"""Serve a local recording interface that launches actual inference on demand."""

import argparse
import functools
import json
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=ROOT.parent)
    parser.add_argument(
        "--output", type=Path, default=ROOT.parent / "runs/final_submission_20261110/demo_record"
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"status": "ready", "events": [], "results": {}, "run_id": None}
    lock = threading.Lock()

    def work():
        with lock:
            state.update(status="running", events=[], results={}, run_id=time.time())
        command = [
            sys.executable,
            str(ROOT / "scripts/run_code_demo.py"),
            "--runtime-root",
            str(args.runtime_root),
            "--output",
            str(args.output),
        ]
        with (args.output / "execution.log").open("w") as log:
            p = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in p.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("[DSP]", "[SVM]", "[CNN]", "[CLAP]", "REAL CODE", "COMPLETE")):
                    with lock:
                        state["events"].append(line.strip())
            code = p.wait()
        with lock:
            state["status"] = "complete" if code == 0 else "error"
            if code == 0:
                state["results"] = {
                    s: json.loads((args.output / f"{s}.json").read_text())
                    for s in ["svm", "cnn", "clap"]
                }
        (args.output / "session.json").write_text(json.dumps(state, indent=2) + "\n")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/api/state":
                with lock:
                    data = json.dumps(state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.split("?")[0] in ["/audio/clean_matched.wav", "/audio/mixture.wav"]:
                file = args.output / self.path.split("?")[0].rsplit("/", 1)[1]
                if not file.exists():
                    self.send_error(404)
                    return
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.split("?")[0] not in ["/", "/index.html"]:
                self.send_error(404)
                return
            self.path = "/index.html"
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/run":
                self.send_error(404)
                return
            if self.headers.get("Origin") not in [
                None,
                f"http://127.0.0.1:{args.port}",
                f"http://localhost:{args.port}",
            ]:
                self.send_error(403)
                return
            with lock:
                if state["status"] == "running":
                    self.send_error(409)
                    return
                state["status"] = "running"
            threading.Thread(target=work, daemon=True).start()
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b'{"started":true}')

        def log_message(self, *args):
            pass

    print(f"Real inference demonstration: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT / "demo"))
    ).serve_forever()


if __name__ == "__main__":
    main()
