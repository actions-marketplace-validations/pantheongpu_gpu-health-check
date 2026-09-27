# Pantheon GPU Health Check

A GPU can be slow, hot or losing bits and still report no error. A job on a
self-hosted runner then trains for six hours on a card that was throttled from
the first minute, or fails in a way nobody can reproduce.

This action runs a short stress test on the runner's GPUs before your job uses
them and gives each card a verdict: **HEALTHY**, **WATCH** or **FAULT**. A
faulty card fails the job. The numbers go into the job summary and the full
reports are kept as an artifact.

The tests are run by [Pantheon](https://pantheongpu.com), an open-source GPU
diagnostics suite for NVIDIA and AMD cards.

## Use it

```yaml
jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: pantheongpu/gpu-health-check@v0
      - uses: actions/checkout@v4
      - run: python train.py
```

With the defaults this takes about five minutes on a runner that has used the
action before, and about ten on the first run, when Pantheon compiles its
workloads for the card.

## What the verdict means

| Verdict | What was found | Job |
|---|---|---|
| HEALTHY | Every workload completed, no errors, temperatures in range. | passes |
| WATCH | The card was thermally throttled, reached 90 C, its memory reached 95 C, it logged correctable errors, or a workload did not complete. | passes, unless `fail-on: watch` |
| FAULT | A memory test failed or the card logged uncorrectable errors. | fails |
| INCOMPLETE | Nothing ran, so nothing is known about the card. | fails |
| NO GPU TESTED | Pantheon found no GPU with a compiler and fell back to its CPU backend. | fails |

A check that could not run fails the job. A green tick has to mean a card was
tested.

## Inputs

| Input | Default | |
|---|---|---|
| `tests` | `baseline_metrics memory_read march_test memory_retention tensor_virus` | Workloads or suites, separated by spaces or commas. `diagnostics` runs every memory test; `all` runs everything. |
| `duration` | `60` | Seconds per workload. |
| `gpu` | `all` | GPU ids, such as `0,1`, or `all`. |
| `mem` | `99` | Percentage of free GPU memory the workloads may use. |
| `fail-on` | `fault` | `fault`, `watch` or `never`. |
| `platform` | `auto` | `auto`, `cuda`, `hip` or `mock`. |
| `version` | `1.2.2` | The Pantheon release to install. |
| `upload-report` | `true` | Keep the JSON reports as a workflow artifact. |
| `artifact-name` | `pantheon-gpu-report` | Change it when a job runs the action more than once. |

## Outputs

| Output | |
|---|---|
| `verdict` | The worst verdict across the tested GPUs. |
| `platform` | `cuda`, `hip`, `mock` or `unknown`. |
| `report-dir` | Directory holding the JSON reports. |

## What the runner needs

- A GPU with its driver, and the compiler for it on the `PATH`: `nvcc` from the
  CUDA toolkit for NVIDIA, `hipcc` from ROCm for AMD. Pantheon compiles its
  workloads for the exact card it finds.
- `python3` with the `venv` module, `make` and `g++`.

GitHub's hosted runners have no GPU. This action is for self-hosted runners and
for GPU runners you rent.

## More examples

Fail on anything short of healthy, and check one card for longer:

```yaml
      - uses: pantheongpu/gpu-health-check@v0
        with:
          gpu: '0'
          duration: 300
          fail-on: watch
```

Check a fleet every night and keep going when one runner is bad:

```yaml
on:
  schedule:
    - cron: '0 3 * * *'
jobs:
  check:
    strategy:
      fail-fast: false
      matrix:
        runner: [gpu-01, gpu-02, gpu-03]
    runs-on: [self-hosted, '${{ matrix.runner }}']
    steps:
      - uses: pantheongpu/gpu-health-check@v0
        with:
          tests: diagnostics
          artifact-name: report-${{ matrix.runner }}
```

Use the verdict in a later step:

```yaml
      - id: gpu
        uses: pantheongpu/gpu-health-check@v0
        with:
          fail-on: never
      - if: steps.gpu.outputs.verdict != 'HEALTHY'
        run: echo "Running on a card marked ${{ steps.gpu.outputs.verdict }}"
```

Try the action on a hosted runner, with no GPU. It exercises the action and
tests no hardware:

```yaml
    runs-on: ubuntu-24.04
    steps:
      - uses: pantheongpu/gpu-health-check@v0
        with:
          platform: mock
          duration: 5
```

## What is in the reports

The artifact holds Pantheon's JSON reports and a summary CSV. They record the
card's name, UUID and serial number, the driver and toolkit versions, the
operating system and kernel, and the CPU and memory size of the host. They stay
in your repository's artifacts. Nothing is sent anywhere else.

If you want a card's results in the public database at
[pantheongpu.com](https://pantheongpu.com), post the report in
[Discussions](https://github.com/pantheongpu/pantheon/discussions).

## Versions

`@v0` follows the latest `v0.x.y` release. Each release pins the Pantheon
release it was tested with, so the checks do not change under a workflow until
the action does. Pin a full version, such as `@v0.1.0`, to hold everything
still.

## License

Apache-2.0.
