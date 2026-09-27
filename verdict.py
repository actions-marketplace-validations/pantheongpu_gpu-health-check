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


def assess_gpu(rows, gpu_id, gpu_name):
    mine = [r for r in rows if r.get("GPU ID") == gpu_id]
    faults, notes = [], []
    throttled, hot, hot_memory, incomplete = [], [], [], []
    ras_serious, benign_ras = {}, []
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

        limit = str(row.get("Limit Reason", "") or "")
        tmax = _num(row.get("Max Temp (C)"))
        tmem = _num(row.get("Max Mem Temp (C)"))
        if limit.lower() == "thermal":
            throttled.append((test, tmax))
        elif tmax >= THERMAL_WATCH_C:
            hot.append((test, tmax))
        if tmem >= MEMORY_THERMAL_WATCH_C:
            hot_memory.append((test, tmem))

    watches = []
    if throttled:
        throttled.sort(key=lambda t: -t[1])
        watches.append("thermal: " + ", ".join(f"{t} thermally throttled, GPU at {c:.0f} C" for t, c in throttled))
    if hot:
        hot.sort(key=lambda t: -t[1])
        watches.append("hot: " + ", ".join(f"{t} GPU reached {c:.0f} C" for t, c in hot))
    if hot_memory:
        hot_memory.sort(key=lambda t: -t[1])
        watches.append("hot memory: " + ", ".join(f"{t} memory reached {c:.0f} C" for t, c in hot_memory))
    if ras_serious:
        totals = {}
        for details in ras_serious.values():
            for item in details:
                name, _, delta = item.rpartition(" ")
                short = name.split(".")[-2] + "." + name.split(".")[-1] if name.count(".") >= 2 else name
                totals[short] = totals.get(short, 0) + _num(delta.lstrip("+"))
        watches.append(f"correctable errors on {len(ras_serious)} workload(s): "
                       + ", ".join(f"{k} +{v:g}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))
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
        notes.append("no workload beyond the idle baseline completed")

    bits = []
    if faults:
        bits.append(f"{len(faults)} fault(s)")
    if throttled:
        bits.append(f"thermally throttled on {len(throttled)}")
    if hot or hot_memory:
        bits.append(f"hot on {len(hot) + len(hot_memory)}")
    if ras_serious:
        bits.append(f"correctable errors on {len(ras_serious)}")
    if incomplete:
        bits.append(f"{len(incomplete)} incomplete")
    if verdict == "HEALTHY":
        bits.append(f"{ran} workload(s) completed")
    return {"gpu_id": gpu_id, "gpu_name": gpu_name, "verdict": verdict,
            "summary": "; ".join(bits), "reasons": faults + watches, "notes": notes,
            "workloads_completed": ran}


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
