"""Exercise concurrent model downloads through real processes and local HTTP."""

import hashlib
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest


REVISION = "a" * 40
ALIAS_WORKER = """
import hashlib, os, sys
from pathlib import Path
from cdskit.localize_models import PRETRAINED_LOCALIZE_MODELS, resolve_localize_model_path
PRETRAINED_LOCALIZE_MODELS['concurrent'] = {
    'name': 'concurrent', 'version': 'v1', 'filename': 'model.pt',
    'url': sys.argv[1] + '/model.pt', 'published': True,
    'sha256': hashlib.sha256(b'complete model').hexdigest(),
}
print('ready', flush=True)
sys.stdin.readline()
p = Path(resolve_localize_model_path('concurrent'))
assert p.read_bytes() == b'complete model'
print(str(p), flush=True)
"""
ESM_WORKER = """
import sys
from cdskit.localize_multilabel_plm import ResidueEncoder
# Load Python dependencies before synchronizing the simultaneous downloads.
import torch, transformers
print('ready', flush=True)
sys.stdin.readline()
encoder = ResidueEncoder({
    'model_name': 'test/esm', 'revision': 'a' * 40,
    'window': 8, 'overlap': 2,
}, 'cpu')
encoder._load()
assert encoder.model.config.hidden_size == 8
assert encoder.tokenizer is not None
print('loaded', flush=True)
"""


@pytest.fixture
def download_server():
    class State:
        def __init__(self):
            self.files = {"model.pt": b"complete model"}
            self.counts = Counter()
            self.mutex = threading.Lock()
            self.weights_heads = None
            self.interrupt_first = False
            self.first_started = threading.Event()
            self.release_first = threading.Event()

    state = State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, body):
            name = unquote(self.path).rsplit("/", 1)[-1]
            content = state.files.get(name)
            if content is None:
                self.send_response(404)
                self.send_header("X-Error-Code", "EntryNotFound")
                self.end_headers()
                return
            if not body and name == "model.safetensors":
                state.weights_heads.wait(timeout=30)
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("ETag", '"' + hashlib.sha256(content).hexdigest() + '"')
            self.send_header("X-Repo-Commit", REVISION)
            self.end_headers()
            if body:
                with state.mutex:
                    state.counts[name] += 1
                    count = state.counts[name]
                if state.interrupt_first and count == 1:
                    state.first_started.set()
                    state.release_first.wait(timeout=30)
                # Keep an actual transfer in flight while the other worker runs.
                time.sleep(0.2)
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        state.release_first.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def download_workers(tmp_path):
    processes = []

    def start(script, endpoint, count=2):
        env = dict(os.environ)
        for key in (
            "CDSKIT_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "TRANSFORMERS_CACHE",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        ):
            env.pop(key, None)
        env.update(
            CDSKIT_MODEL_DIR=str(tmp_path / "cdskit-cache"),
            HF_HOME=str(tmp_path / "hf-home"),
            HF_HUB_CACHE=str(tmp_path / "hf-cache"),
            HF_ENDPOINT=endpoint,
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_XET="1",
            HF_HUB_DISABLE_PROGRESS_BARS="1",
            HF_HUB_ETAG_TIMEOUT="30",
            OMP_NUM_THREADS="1",
        )
        batch = []
        for _ in range(count):
            process = subprocess.Popen(
                [sys.executable, "-c", script, endpoint],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            batch.append(process)
        for process in batch:
            assert process.stdout.readline().strip() == "ready"
        for process in batch:
            process.stdin.write("go\n")
            process.stdin.flush()
        return batch

    yield start
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def finish(process):
    output, error = process.communicate(timeout=60)
    assert process.returncode == 0, error
    return output.strip()


@pytest.mark.subprocess
def test_alias_simultaneous_download_is_shared(download_server, download_workers):
    state, endpoint = download_server
    workers = download_workers(ALIAS_WORKER, endpoint)
    paths = [finish(worker) for worker in workers]
    assert paths[0] == paths[1]
    assert state.counts == {"model.pt": 1}
    assert not list(Path(paths[0]).parent.glob("*.download.*.tmp"))


@pytest.mark.subprocess
def test_alias_killed_downloader_does_not_block_retry(
    download_server, download_workers
):
    state, endpoint = download_server
    state.interrupt_first = True
    first = download_workers(ALIAS_WORKER, endpoint, count=1)[0]
    assert state.first_started.wait(timeout=20)
    first.kill()
    first.communicate(timeout=10)
    state.release_first.set()
    result = finish(download_workers(ALIAS_WORKER, endpoint, count=1)[0])
    assert Path(result).read_bytes() == b"complete model"
    assert state.counts == {"model.pt": 2}


@pytest.mark.ml
@pytest.mark.subprocess
def test_esm_download_uses_shared_hub_locks(
    tmp_path, download_server, download_workers
):
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    # A tiny real ESM checkpoint uses the production ResidueEncoder download path.
    model_dir = tmp_path / "tiny-esm"
    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text("<cls>\n<pad>\n<eos>\n<unk>\nL\nA\nG\n<mask>\n")
    EsmTokenizer(vocab_file=str(vocab)).save_pretrained(model_dir)
    EsmModel(
        EsmConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            pad_token_id=1,
            mask_token_id=7,
        ),
        add_pooling_layer=False,
    ).save_pretrained(model_dir)
    state, endpoint = download_server
    state.files = {p.name: p.read_bytes() for p in model_dir.iterdir() if p.is_file()}
    state.weights_heads = threading.Barrier(2)
    workers = download_workers(ESM_WORKER, endpoint)
    assert [finish(worker) for worker in workers] == ["loaded", "loaded"]
    assert state.counts["model.safetensors"] == 1
    assert state.counts["config.json"] == 1
