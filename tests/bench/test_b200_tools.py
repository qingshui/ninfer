from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.bench import b200_report, b200_sanity

ARTIFACT_ID = "a" * 64


class FakeServe:
    """OpenAI-style streaming endpoint that answers every request with fixed choice texts."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"\n")  # heartbeat
                for index, text in enumerate(outer.texts):
                    event = {"choices": [{"index": index, "delta": {"content": text}}]}
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 3}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeServe":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def write_sanity_inputs(tmp_path: Path, total: int = 30) -> tuple[Path, Path]:
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(json.dumps([{"role": "user", "content": f"问题{i}"}], ensure_ascii=False) + "\n" for i in range(total)),
        encoding="utf-8",
    )
    results = [
        {"qid": qid, "success": True, "generated_text": ["您是检测室内空气吗", "是需要检测环境吗", "您好", "请问需要什么"]}
        for qid in range(1, total + 1)
    ]
    sglang = tmp_path / "benchmark.json"
    sglang.write_text(json.dumps({"summary": {}, "results": results}, ensure_ascii=False), encoding="utf-8")
    return data, sglang


def run_sanity(tmp_path: Path, serve: FakeServe, count: int = 5) -> dict:
    data, sglang = write_sanity_inputs(tmp_path)
    out = tmp_path / "sanity.json"
    code = b200_sanity.main(
        [
            "--ninfer-url", serve.url, "--model", "m", "--sglang-results", str(sglang),
            "--data", str(data), "--count", str(count), "--seed", "7", "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_sanity_samples_fixed_prompts_with_benchmark_sampling_and_passes_clean_output(tmp_path):
    texts = ["您是检测室内空气吗", "是需要检测环境吗", "您好", "请问还需要什么"]
    with FakeServe(texts) as serve:
        first = run_sanity(tmp_path, serve)
        sent = list(serve.requests)
        second = run_sanity(tmp_path, serve)
    assert first["sampling"] == {"temperature": 0.9, "top_p": 0.9, "max_tokens": 15, "n": 4, "enable_thinking": False}
    assert first["sglang_reference"] == "recorded_t1_benchmark"
    assert first["summary"] == {"count": 5, "pass": 5, "needs_review": 0}
    entry = first["prompts"][0]
    assert set(entry) == {"qid", "ninfer_text", "sglang_text", "checks", "verdict"}
    assert entry["ninfer_text"] == texts and len(entry["sglang_text"]) == 4
    assert set(entry["checks"]) == set(b200_sanity.CHECK_NAMES) and entry["verdict"] == "pass"
    assert [p["qid"] for p in first["prompts"]] == [p["qid"] for p in second["prompts"]]
    request = sent[0]
    assert request["n"] == 4 and request["temperature"] == 0.9 and request["top_p"] == 0.9
    assert request["max_tokens"] == 15 and request["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("texts", "failed_check"),
    [
        (["您是检测室内空气吗", "是需要检测环境吗", "您好", ""], "nonempty"),
        (["您是检测室内空气吗", "是需要检测\ufffd环境吗", "您好", "请问"], "no_replacement_char"),
        (["您是检测室内空气吗", "好的好的好的好的好的好的", "您好", "请问"], "no_repetition_collapse"),
        (["Здравствуйте, что", "Нужно проверить", "Да, конечно", "Хорошо"], "script_consistent"),
    ],
)
def test_sanity_marks_bad_output_needs_review(tmp_path, texts, failed_check):
    with FakeServe(texts) as serve:
        document = run_sanity(tmp_path, serve, count=2)
    assert document["summary"] == {"count": 2, "pass": 0, "needs_review": 2}
    entry = document["prompts"][0]
    assert entry["verdict"] == "needs_review" and entry["checks"][failed_check] is False


def write_report_inputs(tmp_path: Path, *, ppl: float = 3.87, throughput: float = 1.7165, duration: float = 1165.15):
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"x")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "request_throughput": throughput, "completed": 2000, "failed": 0, "mean_ttft_ms": 2880.0,
                "p99_ttft_ms": 41414.0, "mean_tpot_ms": 2363.9, "mean_e2el_ms": 75206.0,
                "duration_s": duration, "sglang_baseline_req_throughput": 61.0,
            }
        )
    )  # fmt: skip
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "overall": {"perplexity": ppl},
                "artifact": {"path": str(artifact), "prefill_signature": ARTIFACT_ID},
            }
        )
    )
    return artifact, metrics, report


def run_report(tmp_path: Path, monkeypatch, **inputs) -> tuple[int, dict | None]:
    monkeypatch.setattr(b200_report, "gpu_name", lambda index: "Test GPU")
    artifact, metrics, report = write_report_inputs(tmp_path, **inputs)
    out_json = tmp_path / "out" / "report.json"
    code = b200_report.main(
        [
            "--metrics", str(metrics), "--perplexity-report", str(report),
            "--ppl-ref", str(tmp_path / "ref" / "ppl_ref.json"), "--artifact", str(artifact),
            "--build-dir", str(tmp_path / "build" / "b200-release"), "--gpu", "1",
            "--serve-args", "--max-context 2042 --kv-dtype fp8",
            "--out-json", str(out_json), "--out-md", str(tmp_path / "out" / "report.md"),
        ]
    )  # fmt: skip
    return code, json.loads(out_json.read_text()) if out_json.exists() else None


def test_report_initialises_ppl_ref_then_compares_within_one_percent(tmp_path, monkeypatch):
    code, first = run_report(tmp_path, monkeypatch)
    assert code == 0
    assert first["ppl_ref_status"] == "initialized_from_this_run"
    assert first["ppl_ref"] is None and first["ppl_delta_pct"] is None and first["ppl_within_1pct"] is None
    assert first["ppl"] == 3.87 and first["throughput_bound_ok"] is True
    assert first["ratio_vs_baseline"] == pytest.approx(1.7165 / 61.0)
    assert first["gpu"] == {"index": 1, "name": "Test GPU"}
    assert first["build"]["preset"] == "b200-release" and isinstance(first["build"]["dirty"], bool)
    assert len(first["build"]["git_commit"]) == 40
    assert first["artifact"]["artifact_id"] == ARTIFACT_ID and first["serve_args"].startswith("--max-context")
    assert first["tuning_target"] == {"request_throughput_min": 91.5, "status": "owned_by_tuning"}
    assert "35" in first["locked_ppl_gate_note"]
    assert (tmp_path / "out" / "report.md").read_text().startswith("# b200")

    code, second = run_report(tmp_path, monkeypatch, ppl=3.88)
    assert code == 0 and second["ppl_ref_status"] == "recorded" and second["ppl_ref"] == 3.87
    assert second["ppl_within_1pct"] is True and second["ppl_delta_pct"] == pytest.approx(0.2584, abs=1e-3)

    code, third = run_report(tmp_path, monkeypatch, ppl=3.95)
    assert code == 1 and third["ppl_within_1pct"] is False


@pytest.mark.parametrize("ppl", [1.0, 20.0, float("nan")])
def test_report_rejects_ppl_outside_open_range(tmp_path, monkeypatch, ppl):
    code, report = run_report(tmp_path, monkeypatch, ppl=ppl)
    assert code == 1 and report is None
    assert not (tmp_path / "ref" / "ppl_ref.json").exists()


def test_report_flags_throughput_above_dataset_bound(tmp_path, monkeypatch):
    code, report = run_report(tmp_path, monkeypatch, throughput=2.5, duration=1165.15)
    assert code == 1 and report["throughput_bound_ok"] is False
    code, report = run_report(tmp_path, monkeypatch, throughput=2000 / 1165.15, duration=1165.15)
    assert code == 0 and report["throughput_bound_ok"] is True
