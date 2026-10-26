from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
INGEST_SCRIPT = ROOT / "scripts" / "ingest.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate.py"


def _run_script(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _write_offline_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        f"vector_store:\n  persist_dir: {root / 'chroma'}\n"
        "retrieval:\n  strategy: sparse\n"
        "trace:\n  enabled: false\n",
        encoding="utf-8",
    )


def _ingest_single_document(root: Path, index_dir: Path) -> None:
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "Evaluation gate uses sparse retrieval for deterministic regression.",
        encoding="utf-8",
    )
    completed = _run_script(
        INGEST_SCRIPT,
        str(docs),
        "--index-dir",
        str(index_dir),
        "--embedding-provider",
        "fake",
        cwd=root,
    )
    assert completed.returncode == 0, completed.stderr


def _write_default_golden_test_set(root: Path) -> None:
    path = root / "tests" / "fixtures" / "golden_test_set.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {
                    "query": "deterministic regression",
                    "expected_chunk_ids": ["default::guide-md::0"],
                    "expected_sources": ["guide.md"],
                    "reference_answer": "",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_evaluate_script_uses_default_test_set_and_prints_human_summary(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    _write_default_golden_test_set(tmp_path)

    completed = _run_script(
        EVALUATE_SCRIPT,
        "--index-dir",
        str(index_dir),
        "--strategy",
        "sparse",
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert "hit_rate" in completed.stdout
    assert "mrr" in completed.stdout
    assert "deterministic regression" in completed.stdout
    assert "default::guide-md::0" in completed.stdout
    assert "Evaluation gate uses sparse retrieval" not in completed.stdout


def test_evaluate_script_json_override_outputs_pretty_report(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    custom_test_set = tmp_path / "custom-golden.json"
    custom_test_set.write_text(
        json.dumps(
            [
                {
                    "query": "deterministic regression",
                    "expected_chunk_ids": ["default::guide-md::0"],
                    "expected_sources": ["guide.md"],
                    "reference_answer": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = _run_script(
        EVALUATE_SCRIPT,
        "--test-set",
        str(custom_test_set),
        "--index-dir",
        str(index_dir),
        "--top-k",
        "3",
        "--strategy",
        "sparse",
        "--json",
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("{\n")
    report = json.loads(completed.stdout)
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 1.0
    assert report["per_query"][0]["query"] == "deterministic regression"
    assert "\\u" not in completed.stdout


def test_evaluate_script_threshold_failure_keeps_report_on_stdout(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    _write_default_golden_test_set(tmp_path)

    completed = _run_script(
        EVALUATE_SCRIPT,
        "--index-dir",
        str(index_dir),
        "--strategy",
        "sparse",
        "--min-mrr",
        "1.1",
        "--json",
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    assert json.loads(completed.stdout)["mrr"] == 1.0
    assert "评估门禁失败" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_evaluate_script_failed_human_summary_shows_expected_source(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    failed_test_set = tmp_path / "failed-golden.json"
    failed_test_set.write_text(
        json.dumps(
            [
                {
                    "query": "deterministic regression",
                    "expected_chunk_ids": ["default::missing-md::0"],
                    "expected_sources": ["missing.md"],
                    "reference_answer": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    completed = _run_script(
        EVALUATE_SCRIPT,
        "--test-set",
        str(failed_test_set),
        "--index-dir",
        str(index_dir),
        "--strategy",
        "sparse",
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    assert "[FAIL]" in completed.stdout
    assert "default::guide-md::0" in completed.stdout
    assert "expected_sources=missing.md" in completed.stdout
    assert "Evaluation gate uses sparse retrieval" not in completed.stdout
    assert "评估门禁失败" in completed.stderr


def test_evaluate_script_reports_bad_golden_set_as_user_error(tmp_path: Path):
    _write_offline_config(tmp_path)
    invalid_test_set = tmp_path / "invalid.json"
    invalid_test_set.write_text("[]", encoding="utf-8")

    completed = _run_script(EVALUATE_SCRIPT, "--test-set", str(invalid_test_set), cwd=tmp_path)

    assert completed.returncode != 0
    assert "错误" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_evaluate_script_reports_query_config_error_as_user_error(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    _write_default_golden_test_set(tmp_path)
    (tmp_path / "config" / "settings.yaml").write_text(
        f"vector_store:\n  persist_dir: {tmp_path / 'chroma'}\n"
        "retrieval:\n  strategy: unsupported\n"
        "trace:\n  enabled: false\n",
        encoding="utf-8",
    )

    completed = _run_script(EVALUATE_SCRIPT, "--index-dir", str(index_dir), cwd=tmp_path)

    assert completed.returncode != 0
    assert "错误" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_evaluate_script_rejects_invalid_top_k_as_user_error(tmp_path: Path):
    _write_offline_config(tmp_path)
    index_dir = tmp_path / "index"
    _ingest_single_document(tmp_path, index_dir)
    _write_default_golden_test_set(tmp_path)

    completed = _run_script(
        EVALUATE_SCRIPT,
        "--index-dir",
        str(index_dir),
        "--top-k",
        "0",
        cwd=tmp_path,
    )

    assert completed.returncode != 0
    assert "错误" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_evaluate_script_keeps_unexpected_config_parse_errors_visible(tmp_path: Path):
    _write_default_golden_test_set(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("retrieval: [", encoding="utf-8")

    completed = _run_script(EVALUATE_SCRIPT, cwd=tmp_path)

    assert completed.returncode != 0
    assert "Traceback" in completed.stderr


def test_evaluate_script_rejects_unknown_strategy(tmp_path: Path):
    completed = _run_script(EVALUATE_SCRIPT, "--strategy", "unknown", cwd=tmp_path)

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
