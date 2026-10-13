#!/usr/bin/env python3
"""Turn Pantheon's JSON reports into one verdict per GPU.

The rules mirror `assess_gpu` in Pantheon's own pantheon.py, so a run gated by
this action and a run read by hand reach the same word. They live here as well
because a Pantheon release older than the verdict does not put it in the report.
"""
import argparse
import glob
import json
import os
import re
import sys

DIAGNOSTIC_TESTS = {
    "march_test", "galpat", "memory_hammer", "memory_retention",
    "memory_retention_bake", "ras_validator",
}
THERMAL_WATCH_C = 90
MEMORY_THERMAL_WATCH_C = 95
# The PCIe link cycling power states ticks this counter on healthy hardware.
BENIGN_RAS_TOKENS = ("l0_to_recovery",)
SEVERITY = {"HEALTHY": 0, "WATCH": 1, "INCOMPLETE": 2, "FAULT": 3, "NO GPU TESTED": 4}
NO_GPU = "NO GPU TESTED"


def _num(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def split_ras_details(delta_text):
    benign, serious = [], []
    for token in str(delta_text or "").split("||"):
        token = token.strip()
        if not token or token in ("None", "N/A"):
            continue
        (benign if any(b in token for b in BENIGN_RAS_TOKENS) else serious).append(token)
    return benign, serious


def limit_reason_set(value):
    """The lower-case labels of a Limit Reason such as "Power|Thermal"."""
    return {part.strip().lower() for part in str(value or "").split("|") if part.strip()}


def is_skipped_row(row):
    """True for a row of a workload that declined to run, which is not a pass."""
    return str(row.get("Status") or "").upper() == "SKIP" or row.get("Unit") == "SKIP"


def assess_gpu(rows, gpu_id, gpu_name):
    """Turn one GPU's result rows into a verdict with its evidence.

    Findings are grouped by kind, so a card that is hot on five workloads and
    slow on nine reads as two lines, worst first, rather than fourteen in the
    order the workloads happened to run.
    """
    mine = [r for r in rows if r.get("GPU ID") == gpu_id]
    faults, notes = [], []
    throttled, hot, hot_memory, incomplete = [], [], [], []
    ras_serious, benign_ras = {}, []
    skipped = []
    throttle_seconds = {}
    ran = 0
    for row in mine:
        test = row.get("Test Name", "?")
        if row.get("Failure Stage"):
            notes.append(f"{test} did not run ({row.get('Failure Stage')}: {row.get('Failure Reason')})")
            continue
        unit = row.get("Unit")
        failed = unit == "ERR" or str(row.get("Status") or "PASS").upper() == "FAIL"
        if failed:
            if test in DIAGNOSTIC_TESTS:
                faults.append(f"{test} failed: memory errors detected or the workload aborted, see its log")
            else:
                incomplete.append(test)
            continue
        if is_skipped_row(row):
            # It declined to run; there is no measurement and nothing to judge.
            skipped.append(test)
            reason = row.get("Skip Reason")
            notes.append(f"{test} was skipped" + (f" ({reason})" if reason else ""))
            continue
        if test != "baseline_metrics":
            ran += 1

        ras_status = str(row.get("RAS Status", "")).upper()
        if ras_status == "ERROR":
            faults.append(f"{test}: uncorrectable errors ({row.get('RAS Error Delta')})")
        elif ras_status == "WARNING":
            benign, serious = split_ras_details(row.get("RAS Error Delta"))
            if serious:
                ras_serious[test] = serious
            if benign:
                benign_ras.append(test)

        # The reason is the most common label of the samples, joined with "|"
        # when a sample had several ("Power|Thermal"), so it is read as a set.
        reasons = limit_reason_set(row.get("Limit Reason"))
        tmax = _num(row.get("Max Temp (C)"))
        tmem = _num(row.get("Max Mem Temp (C)"))
        throttle_s = _num(row.get("Throttle Time (s)"))
        if "thermal" in reasons:
            throttled.append((test, tmax))
            throttle_seconds[test] = throttle_s
        elif tmax >= THERMAL_WATCH_C:
            hot.append((test, tmax))
        if throttle_s > 0 and not (reasons - {"none", "n/a", "idle"}):
            # Throttled for part of the run, but the commonest label of the
            # samples is "None", so the cause was not recorded. Not a finding
            # (power-cap throttling is normal under load), but not silent.
            notes.append(f"{test}: throttled for {throttle_s:.0f} s, cause not recorded")
        if tmem >= MEMORY_THERMAL_WATCH_C:
            hot_memory.append((test, tmem))

    watches = []
    if throttled:
        throttled.sort(key=lambda t: -t[1])
        watches.append("thermal: " + ", ".join(
            f"{t} thermally throttled, GPU at {c:.0f} C"
            + (f", {throttle_seconds[t]:.0f} s throttled" if throttle_seconds.get(t, 0) > 0 else "")
            for t, c in throttled))
    if hot:
        hot.sort(key=lambda t: -t[1])
        watches.append("hot: " + ", ".join(f"{t} GPU reached {c:.0f} C" for t, c in hot))
    if hot_memory:
        hot_memory.sort(key=lambda t: -t[1])
        watches.append("hot memory: " + ", ".join(f"{t} memory reached {c:.0f} C" for t, c in hot_memory))
    if ras_serious:
        totals, events = {}, []
        for details in ras_serious.values():
            for item in details:
                counted = re.match(r"^(.*\S)\s+\+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$", item)
                if not counted:
                    # An event line, not a counter: keep its words, not a "+0".
                    if item not in events:
                        events.append(item if len(item) <= 120 else item[:117] + "...")
                    continue
                name = counted.group(1)
                short = name.split(".")[-2] + "." + name.split(".")[-1] if name.count(".") >= 2 else name
                totals[short] = totals.get(short, 0) + _num(counted.group(2))
        parts = [f"{k} +{v:g}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])] + events
        watches.append(f"correctable errors on {len(ras_serious)} workload(s): " + ", ".join(parts))
    if incomplete:
        watches.append("did not complete: " + ", ".join(incomplete))

    if benign_ras:
        notes.append(f"PCIe link recovery events on {len(benign_ras)} workload(s): "
                     "link power-state cycling, not a fault")
    if faults:
        verdict = "FAULT"
    elif watches:
        verdict = "WATCH"
    elif ran:
        verdict = "HEALTHY"
    else:
        verdict = "INCOMPLETE"
        if skipped:
            notes.append(f"nothing ran: {len(skipped)} workload(s) skipped ({', '.join(skipped)})")
        else:
            notes.append("nothing ran: no workload beyond the idle baseline completed")

    summary_bits = []
    if faults:
        summary_bits.append(f"{len(faults)} fault(s)")
    if throttled:
        summary_bits.append(f"thermally throttled on {len(throttled)}")
    if hot or hot_memory:
        summary_bits.append(f"hot on {len(hot) + len(hot_memory)}")
    if ras_serious:
        summary_bits.append(f"correctable errors on {len(ras_serious)}")
    if incomplete:
        summary_bits.append(f"{len(incomplete)} incomplete")
    if skipped and verdict != "INCOMPLETE":
        summary_bits.append(f"{len(skipped)} skipped")
    if verdict == "INCOMPLETE" and skipped:
        summary_bits.append("nothing ran")
    if verdict == "HEALTHY":
        summary_bits.append(f"{ran} workload(s) completed")

    return {
        "gpu_id": gpu_id, "gpu_name": gpu_name, "verdict": verdict,
        "summary": "; ".join(summary_bits),
        "reasons": faults + watches, "notes": notes,
        "findings": {
            "faults": faults,
            "throttled": [{"test": t, "temp_c": c} for t, c in throttled],
            "hot": [{"test": t, "temp_c": c} for t, c in hot],
            "hot_memory": [{"test": t, "temp_c": c} for t, c in hot_memory],
            "correctable_errors": ras_serious,
            "incomplete": incomplete,
            "skipped": skipped,
        },
        "workloads_completed": ran,
    }


def load_reports(report_dir):
    """Rows and GPUs from every report in the directory, each workload once.

    Pantheon writes a session report and, for each workload that completed, a
    twin holding the same row. Only the twin carries a run id, and a workload
    that failed appears in the session report alone, so every file is read and
    a row is counted once by its content.
    """
    rows, gpus, platforms, seen = [], {}, set(), set()
    for path in sorted(glob.glob(os.path.join(report_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        for gpu in report.get("gpu_static_info") or []:
            if not isinstance(gpu, dict):
                continue
            gpus.setdefault(gpu.get("id"), gpu.get("name") or "GPU")
            kind = str(gpu.get("type") or "").upper()
            platforms.add({"NVIDIA": "cuda", "AMD": "hip", "MOCK": "mock"}.get(kind, "unknown"))
        for row in report.get("test_results") or []:
            if not isinstance(row, dict):
                continue
            key = json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows, gpus, platforms


def requested_ids(requested_gpus, gpus):
    """The GPUs the job asked to have tested.

    A report lists every GPU in the host, tested or not, so the request decides
    which cards are judged: a card nobody asked about is left out, and a card
    that was asked about and produced nothing is INCOMPLETE.
    """
    text = str(requested_gpus or "all").strip().lower()
    if text in ("", "all"):
        return sorted(gpus, key=str)
    ids = []
    for part in text.split(","):
        part = part.strip()
        if part.isdigit() and int(part) not in ids:
            ids.append(int(part))
    return ids


def decide(rows, gpus, platforms, fail_on, requested_platform, requested_gpus="all"):
    platform = "unknown"
    for candidate in ("cuda", "hip", "mock"):
        if candidate in platforms:
            platform = candidate
            break
    assessments = [assess_gpu(rows, gid, gpus.get(gid, "not found on this runner"))
                   for gid in requested_ids(requested_gpus, gpus)]
    messages = []
    if not rows or not assessments:
        overall = "INCOMPLETE"
        messages.append("Pantheon produced no report. The workloads did not run; see the log of the run step.")
    elif platform == "mock" and requested_platform != "mock":
        overall = NO_GPU
        messages.append("Pantheon fell back to its CPU mock backend: it found no GPU with a compiler "
                        "(nvcc for NVIDIA, hipcc for AMD) on this runner. No hardware was tested.")
    else:
        overall = max((a["verdict"] for a in assessments), key=lambda v: SEVERITY[v])
    if fail_on == "never":
        failed = False
    elif fail_on == "watch":
        failed = overall != "HEALTHY"
    else:
        failed = overall in ("FAULT", "INCOMPLETE", NO_GPU)
    return {"verdict": overall, "platform": platform, "failed": failed,
            "assessments": assessments, "messages": messages}


def _cell(value):
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"
    return str(value).replace("|", "/").replace("\n", " ")


def render_summary(result, rows):
    lines = [f"## GPU health check: {result['verdict']}", ""]
    for message in result["messages"]:
        lines += [f"> {message}", ""]
    if result["platform"] == "mock":
        lines += ["> Backend: mock. These numbers come from the CPU and describe no GPU.", ""]
    for a in result["assessments"]:
        tail = f" ({a['summary']})" if a["summary"] else ""
        lines.append(f"**GPU {a['gpu_id']}, {a['gpu_name']}: {a['verdict']}**{tail}")
        lines += [f"- {reason}" for reason in a["reasons"]]
        lines += [f"- note: {note}" for note in a["notes"]]
        lines.append("")
    if rows:
        lines += ["| Workload | GPU | Score | Unit | Max temp (C) | Max power (W) | Limit | RAS |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append("| " + " | ".join(_cell(row.get(k)) for k in (
                "Test Name", "GPU ID", "Score", "Unit", "Max Temp (C)", "Max Power (W)",
                "Limit Reason", "RAS Status")) + " |")
        lines.append("")
    lines.append("Measured with [Pantheon](https://pantheongpu.com), an open-source GPU diagnostics suite.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", required=True)
    parser.add_argument("--fail-on", default="fault", choices=["fault", "watch", "never"])
    parser.add_argument("--requested-platform", default="auto")
    parser.add_argument("--requested-gpus", default="all")
    parser.add_argument("--summary")
    parser.add_argument("--outputs")
    args = parser.parse_args(argv)

    rows, gpus, platforms = load_reports(args.reports)
    result = decide(rows, gpus, platforms, args.fail_on, args.requested_platform, args.requested_gpus)
    text = render_summary(result, rows)
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    if args.outputs:
        with open(args.outputs, "a", encoding="utf-8") as handle:
            handle.write(f"verdict={result['verdict']}\n")
            handle.write(f"platform={result['platform']}\n")
            handle.write(f"failed={'true' if result['failed'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
