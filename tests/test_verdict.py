import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verdict  # noqa: E402


def row(test, **over):
    base = {"Test Name": test, "GPU ID": 0, "Score": 100.0, "Unit": "GB/s", "Status": None,
            "Failure Stage": None, "RAS Status": "CLEAN", "RAS Error Delta": "None",
            "Limit Reason": "None", "Max Temp (C)": 55.0, "Max Mem Temp (C)": "N/A", "Max Power (W)": 200.0}
    base.update(over)
    return base


def write(tmp_path, name, rows, kind="NVIDIA", run_id="r1", gpu_name="NVIDIA Test Card"):
    report = {"run_id": run_id, "gpu_static_info": [{"id": 0, "type": kind, "name": gpu_name}],
              "test_results": rows}
    (tmp_path / name).write_text(json.dumps(report))


def decide(tmp_path, fail_on="fault", requested="auto", gpus_asked="all"):
    rows, gpus, platforms = verdict.load_reports(str(tmp_path))
    return verdict.decide(rows, gpus, platforms, fail_on, requested, gpus_asked), rows


def two_gpu_host(tmp_path, rows):
    report = {"run_id": "r1", "test_results": rows,
              "gpu_static_info": [{"id": 0, "type": "NVIDIA", "name": "A"}, {"id": 1, "type": "NVIDIA", "name": "B"}]}
    (tmp_path / "a.json").write_text(json.dumps(report))


def test_a_card_nobody_asked_about_is_not_judged(tmp_path):
    # The report lists every GPU in the host; only GPU 0 was requested and run.
    two_gpu_host(tmp_path, [row("memory_read")])
    result, _ = decide(tmp_path, gpus_asked="0")
    assert [a["gpu_id"] for a in result["assessments"]] == [0]
    assert result["verdict"] == "HEALTHY" and not result["failed"]


def test_a_requested_card_that_produced_nothing_is_incomplete(tmp_path):
    two_gpu_host(tmp_path, [row("memory_read")])
    result, _ = decide(tmp_path, gpus_asked="all")
    assert [a["verdict"] for a in result["assessments"]] == ["HEALTHY", "INCOMPLETE"]
    assert result["verdict"] == "INCOMPLETE" and result["failed"]


def test_a_requested_card_the_host_does_not_have_is_incomplete(tmp_path):
    two_gpu_host(tmp_path, [row("memory_read")])
    result, _ = decide(tmp_path, gpus_asked="0, 5")
    assert [(a["gpu_id"], a["verdict"]) for a in result["assessments"]] == [(0, "HEALTHY"), (5, "INCOMPLETE")]
    assert result["assessments"][1]["gpu_name"] == "not found on this runner"


def test_healthy_card_passes(tmp_path):
    write(tmp_path, "a.json", [row("baseline_metrics", Score=0.0), row("memory_read"), row("march_test")])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "HEALTHY" and not result["failed"] and result["platform"] == "cuda"


def test_session_report_and_its_twin_count_once(tmp_path):
    # As Pantheon writes them: the session report has no run id, the twin has one.
    write(tmp_path, "pantheon_report_r1.json", [row("memory_read")], run_id=None)
    write(tmp_path, "pantheon_report_r1_0001_memory_read_gpu0.json", [row("memory_read")], run_id="r1")
    result, rows = decide(tmp_path)
    assert len(rows) == 1 and result["assessments"][0]["workloads_completed"] == 1


def test_a_failed_workload_in_the_session_report_is_kept(tmp_path):
    # A workload that failed has no twin; it must not be lost with the duplicates.
    write(tmp_path, "pantheon_report_r1.json",
          [row("memory_read"), row("march_test", Unit="ERR", Score=0.0)], run_id=None)
    write(tmp_path, "pantheon_report_r1_0001_memory_read_gpu0.json", [row("memory_read")], run_id="r1")
    result, rows = decide(tmp_path)
    assert len(rows) == 2 and result["verdict"] == "FAULT"


def test_the_same_workload_run_twice_counts_twice(tmp_path):
    write(tmp_path, "a.json", [row("memory_read", Score=100.0)], run_id="r1")
    write(tmp_path, "b.json", [row("memory_read", Score=101.0)], run_id="r2")
    assert len(decide(tmp_path)[1]) == 2


def test_failed_memory_diagnostic_is_a_fault(tmp_path):
    write(tmp_path, "a.json", [row("memory_read"), row("march_test", Unit="ERR", Score=0.0)])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "FAULT" and result["failed"]


def test_uncorrectable_errors_are_a_fault(tmp_path):
    write(tmp_path, "a.json", [row("memory_read", **{"RAS Status": "ERROR", "RAS Error Delta": "ecc.uncorrected +2"})])
    assert decide(tmp_path)[0]["verdict"] == "FAULT"


def test_thermal_throttling_is_a_watch(tmp_path):
    write(tmp_path, "a.json", [row("memory_read", **{"Limit Reason": "Thermal", "Max Temp (C)": 95.0})])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "WATCH" and not result["failed"]
    assert decide(tmp_path, fail_on="watch")[0]["failed"]


def test_hot_card_is_a_watch(tmp_path):
    write(tmp_path, "a.json", [row("tensor_virus", **{"Max Temp (C)": 91.0})])
    assert decide(tmp_path)[0]["verdict"] == "WATCH"


def test_link_recovery_alone_is_not_held_against_the_card(tmp_path):
    write(tmp_path, "a.json", [row("memory_read", **{"RAS Status": "WARNING",
                                                     "RAS Error Delta": "vendor_ras.pcie.l0_to_recovery +1"})])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "HEALTHY"
    assert any("link recovery" in note for note in result["assessments"][0]["notes"])


def test_correctable_errors_are_a_watch(tmp_path):
    write(tmp_path, "a.json", [row("memory_read", **{"RAS Status": "WARNING",
                                                     "RAS Error Delta": "vendor_ras.pcie.bad_tlp +1972"})])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "WATCH" and "bad_tlp" in result["assessments"][0]["reasons"][0]


def test_only_the_idle_baseline_is_incomplete_and_fails(tmp_path):
    write(tmp_path, "a.json", [row("baseline_metrics", Score=0.0)])
    result, _ = decide(tmp_path)
    assert result["verdict"] == "INCOMPLETE" and result["failed"]


def test_no_reports_fails_rather_than_passing_silently(tmp_path):
    result, _ = decide(tmp_path)
    assert result["verdict"] == "INCOMPLETE" and result["failed"]
    assert not decide(tmp_path, fail_on="never")[0]["failed"]


def test_silent_fallback_to_mock_fails(tmp_path):
    write(tmp_path, "a.json", [row("memory_read")], kind="MOCK", gpu_name="Mock GPU")
    result, _ = decide(tmp_path, requested="auto")
    assert result["verdict"] == verdict.NO_GPU and result["failed"] and result["platform"] == "mock"


def test_mock_that_was_asked_for_passes(tmp_path):
    write(tmp_path, "a.json", [row("memory_read")], kind="MOCK", gpu_name="Mock GPU")
    result, _ = decide(tmp_path, requested="mock")
    assert result["verdict"] == "HEALTHY" and not result["failed"]


def test_worst_card_decides_the_job(tmp_path):
    report = {"run_id": "r1",
              "gpu_static_info": [{"id": 0, "type": "NVIDIA", "name": "A"}, {"id": 1, "type": "NVIDIA", "name": "B"}],
              "test_results": [row("memory_read"), row("march_test", **{"GPU ID": 1, "Unit": "ERR"})]}
    (tmp_path / "a.json").write_text(json.dumps(report))
    result, _ = decide(tmp_path)
    assert [a["verdict"] for a in result["assessments"]] == ["HEALTHY", "FAULT"]
    assert result["verdict"] == "FAULT"


def test_a_corrupt_report_is_skipped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    write(tmp_path, "a.json", [row("memory_read")])
    assert decide(tmp_path)[0]["verdict"] == "HEALTHY"


def test_outputs_and_summary_are_written(tmp_path):
    reports = tmp_path / "db"
    reports.mkdir()
    write(reports, "a.json", [row("memory_read", Score=1234.5)])
    out, summary = tmp_path / "out.txt", tmp_path / "summary.md"
    assert verdict.main(["--reports", str(reports), "--outputs", str(out), "--summary", str(summary)]) == 0
    assert out.read_text().splitlines() == ["verdict=HEALTHY", "platform=cuda", "failed=false"]
    text = summary.read_text()
    assert "GPU health check: HEALTHY" in text and "| memory_read | 0 | 1,234.50 | GB/s |" in text


@pytest.mark.parametrize("value", ["a | b", "line\nbreak"])
def test_table_cells_cannot_break_the_table(value):
    assert "|" not in verdict._cell(value) and "\n" not in verdict._cell(value)
