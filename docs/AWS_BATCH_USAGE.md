# Running training sweeps on AWS

The pipeline runs IsaacLab sweeps on AWS Batch GPU instances. You upload a
sweep YAML to S3, invoke a Lambda dispatcher, watch the job, and pull the
results down. Everything is driven through the Makefile at the repo root.

## Prerequisites

You need:

- AWS configured (`~/.aws/credentials`)
- `jq` and `uuidgen` on your `$PATH` (almost certainly already there)
- The training repo's GitHub deploy key already populated in Secrets Manager
  (the team has set this up once; you don't need to touch it)

## TL;DR

```bash
make submit CONFIG=path/to/sweep.yaml S3_KEY=runs/$(date +%F)/experiment_name/sweep.yaml
make status            # watches every 15s, exits when done
make download_csv      # pulls just the results CSV(s)
```

That's the whole loop. Sections below break down each target.

## The targets, in the order you'll use them

### 1. Upload a sweep config

```bash
make upload CONFIG=configs/sweep.yaml \
            S3_KEY=runs/2026-04-30/curriculum-tweak/sweep.yaml
```

Copies your local YAML to
`s3://ethrc-aeronautics/droneracing/isaac-rl-runs/sweeps/inputs/<S3_KEY>`.

`S3_KEY` is whatever sub-path you want under the `inputs/` prefix — pick
something descriptive so future you can find the config that produced a
given run. Convention: `runs/<date>/<experiment-name>/sweep.yaml`.

### 2. Submit a job

```bash
make submit S3_KEY=runs/2026-04-30/curriculum-tweak/sweep.yaml
```

Or upload + submit in one go (just add `CONFIG=...`):

```bash
make submit CONFIG=configs/sweep.yaml \
            S3_KEY=runs/2026-04-30/curriculum-tweak/sweep.yaml
```

By default the job clones `main` of the training repo. To pin to a branch
or commit:

```bash
make submit S3_KEY=...  BRANCH=feature/curriculum-tweak
make submit S3_KEY=...  COMMIT_SHA=4df6560e187f2cc66685b41b21b259f4485d0c22
```

`COMMIT_SHA` overrides `BRANCH` if both are passed.

The target prints the job ID and the run ID, and saves them to
`.aero-last-job-id` / `.aero-last-run-id` so the next targets pick them up
automatically. You don't have to copy/paste IDs around.

### 3. Watch the status

```bash
make status
```

Refreshes every 15 seconds. Auto-exits when the job hits `SUCCEEDED` or
`FAILED`. `Ctrl+C` to detach early — the job keeps running.

Expected progression on a cold start:

| Status | What's happening | Roughly how long |
|---|---|---|
| `SUBMITTED` → `PENDING` → `RUNNABLE` | Batch queues the job | a few seconds |
| `STARTING` | EC2 instance boots, ECS pulls the container | 2–5 min (custom AMI), 5–10 min (first time) |
| `RUNNING` | Container is alive; entrypoint clones repo, fetches config, runs `sweep.py` | duration of your sweep |
| `SUCCEEDED` / `FAILED` | Done | — |

If a job fails, the status table shows `statusReason`. The recipient configured
in `terraform/terraform.tfvars` (`alert_email`) also gets an email with the
job name and reason.

### 4. Tail the container logs

```bash
make logs
```

Streams the container's stdout/stderr from CloudWatch as it happens. Works
once the job is `RUNNING` (no log stream exists before that).

`Ctrl+C` to stop tailing — again, the job keeps running. You can re-tail
later as long as the job is still alive or hasn't aged out of CloudWatch
retention (30 days by default).

### 5. Cancel a running job (if needed)

If you submitted in error or a sweep is going off the rails:

```bash
make cancel
```

Sends `batch:TerminateJob`. The job transitions to `FAILED` within seconds,
the EC2 instance scales back down, and you'll get the standard failure
notification email. You only pay for the time the instance was actually
running before termination.

To cancel a specific job (not the most-recently-submitted one):

```bash
make cancel JOB_ID=<id>
```

### 6. Download the results

Two flavors. Pick the one that matches what you actually want.

**Just the sweep results CSV** — what you usually want for quick analysis:

```bash
make download_csv
# downloads to ./downloaded/<run_id>/results_<timestamp>.csv
```

**The whole run output dir** — TensorBoard event files, checkpoints, full
skrl logs, sweep CSV(s), `metadata.json`. Larger but complete:

```bash
make download_results
# downloads to ./downloaded/<run_id>/...
```

For both, override the destination with `DEST=...`:

```bash
make download_csv DEST=~/experiments/quad-grid-2026-04-30
```

If you want to look at a different run than the most-recently-submitted one,
pass the run id explicitly:

```bash
make download_csv RUN_ID=run-20260430T162814Z-e8134d
```

## Variables reference

| Variable | Default | Used by | Notes |
|---|---|---|---|
| `CONFIG` | (none) | `upload`, `submit` | Local path to the sweep YAML |
| `S3_KEY` | (none) | `upload`, `submit` | Sub-path under `inputs/` |
| `BRANCH` | `main` | `submit` | Git branch to check out in the container |
| `COMMIT_SHA` | (none) | `submit` | Pinned SHA; overrides `BRANCH` |
| `JOB_ID` | last submitted | `status`, `logs`, `cancel` | Batch job id |
| `RUN_ID` | last submitted | `status`, `download_*` | Run id used in S3 paths |
| `DEST` | `./downloaded/<RUN_ID>` | `download_*` | Local download directory |

The "last submitted" defaults read from `.aero-last-job-id` /
`.aero-last-run-id` in the repo root. These are written by `make submit` and
git-ignored, so each user has their own.

## Full example

A complete run from scratch:

```bash
# 1. Make sure you're using the right AWS profile (only if you have multiple).
export AWS_PROFILE=aeronautics

# 2. Edit your sweep config locally.
$EDITOR configs/sweep.yaml

# 3. Upload + submit in one shot, against your dev branch.
make submit \
  CONFIG=configs/sweep.yaml \
  S3_KEY=runs/$(date +%F)/curriculum-tweak/sweep.yaml \
  BRANCH=feature/curriculum-tweak

# 4. Watch it run.
make status        # blocks until SUCCEEDED/FAILED, ~30 min for a real sweep

# 5. (Optional) tail the live training output in another terminal.
make logs

# 6. Pull the analysis CSV down.
make download_csv

# 7. Inspect.
ls -la downloaded/$(cat .aero-last-run-id)
```

## AWS Setup

The pipeline consists of six AWS pieces wired together:

```
   you / Makefile
        │
        ▼ (1) S3 PUT  - upload sweep yaml
   S3 bucket  <──────── (5) entrypoint reads sweep yaml, writes outputs ───┐
        │                                                                  │
        ▼ (2) Lambda invoke                                                │
   Lambda dispatcher  ── batch:SubmitJob ───> (3) AWS Batch  ───> (4) EC2 (g6e)
                                                       │              container runs
                                                       │              entrypoint.sh
                                                       │              └─ git clone repo
                                                       │              └─ sweep.py
                                                       │
                                                       └─> (6) CloudWatch Logs
```

1. **S3** — `ethrc-aeronautics/droneracing/isaac-rl-runs/sweeps/`. `inputs/` for
   sweep configs, `outputs/runs/<run_id>/` for results.
2. **Lambda** (`aero-batch-dispatcher`) — receives a tiny JSON payload from
   the Makefile, calls `batch:SubmitJob` with the right env vars, returns the
   job id.
3. **AWS Batch** — manages a queue and a GPU compute environment that auto-
   scales from 0 to a few `g6.2xlarge` instances. Sized so it costs $0 when
   idle.
4. **EC2 instance** — boots from a custom AMI that already has the IsaacLab
   Docker image cached, so cold start is ~2–3 min instead of ~10. ECS hands
   the container the env vars and runs `bash /opt/entrypoint.sh`.
5. **The container's entrypoint script** — fetches the GitHub deploy key from
   Secrets Manager, clones the training repo at the requested ref, downloads
   the sweep YAML from S3, runs `python scripts/sweep.py`, and uploads logs
   + results back to S3 on exit (even if `sweep.py` crashed).
6. **CloudWatch Logs** — `awslogs` driver writes the container's stdout to
   `/aws/batch/job` so `make logs` can tail it. 30-day retention.

Failure alerts route through SNS → email
