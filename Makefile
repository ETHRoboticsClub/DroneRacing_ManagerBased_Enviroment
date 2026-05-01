###############################################################################
# Makefile for the IsaacLab AWS Batch pipeline.
#
# Fixed identifiers (do not change here - terraform owns them):
#   bucket / prefix / AWS profile / region / Lambda / job def
#
# The training repo URL is baked into the terraform job definition (var
# `training_repo` in terraform/variables.tf), so we never pass it from here.
#
# Variables you set on the make command line:
#   CONFIG       local sweep yaml file              (for `make upload`/`submit`)
#   S3_KEY       sub-path under <prefix>/inputs/    (required for upload/submit)
#                  e.g.  S3_KEY=runs/2026-04-30/quadcopter-grid/sweep.yaml
#                  resolves to:
#                  s3://<bucket>/<prefix>/inputs/runs/2026-04-30/quadcopter-grid/sweep.yaml
#   BRANCH       git branch to clone                (default: main)
#   COMMIT_SHA   pinned git SHA, overrides BRANCH   (optional)
#   JOB_ID       Batch job id                       (default: last submitted)
#   RUN_ID       run id for output paths            (default: last submitted)
#   DEST         local download directory           (default: ./downloaded/<RUN_ID>)
#
# Examples:
#   make upload   CONFIG=docker/isaaclab/sweep.yaml S3_KEY=runs/foo/sweep.yaml
#   make submit   CONFIG=docker/isaaclab/sweep.yaml S3_KEY=runs/foo/sweep.yaml
#   make submit   S3_KEY=runs/foo/sweep.yaml COMMIT_SHA=4df6560
#   make status                                      # watches every 15s
#   make logs
#   make download_csv                                # just the sweep results
#   make download_results                            # full run output dir
###############################################################################

# --- fixed ---------------------------------------------------------------- #
BUCKET   := ethrc-aeronautics
PREFIX   := droneracing/isaac-rl-runs/sweeps
REGION   := us-east-2
LAMBDA   := aero-batch-dispatcher
JOBDEF   := aero-batch-isaaclab

# --- per-call defaults ---------------------------------------------------- #
BRANCH      ?= main
COMMIT_SHA  ?=
JOB_ID      ?= $(shell cat .aero-last-job-id 2>/dev/null)
RUN_ID      ?= $(shell cat .aero-last-run-id 2>/dev/null)

# --- derived -------------------------------------------------------------- #
INPUT_URI    := s3://$(BUCKET)/$(PREFIX)/inputs/$(S3_KEY)
OUTPUTS_URI   = s3://$(BUCKET)/$(PREFIX)/outputs/runs/$(RUN_ID)
REF          := $(if $(COMMIT_SHA),$(COMMIT_SHA),$(BRANCH))

.DEFAULT_GOAL := help
.PHONY: help upload submit status logs cancel download_results download_csv

# --------------------------------------------------------------------------- #

help:
	@echo "Targets:"
	@echo "  upload             upload a local sweep yaml to <prefix>/inputs/<S3_KEY>"
	@echo "                       CONFIG=<path> S3_KEY=<sub-path>"
	@echo "  submit             start a Batch job (uploads first if CONFIG is given)"
	@echo "                       S3_KEY=<sub-path> [BRANCH=...] [COMMIT_SHA=...]"
	@echo "  status             watch job status + output URI (15s refresh)"
	@echo "                       [JOB_ID=...]"
	@echo "  logs               tail CloudWatch logs"
	@echo "                       [JOB_ID=...]"
	@echo "  cancel             terminate a running job"
	@echo "                       [JOB_ID=...]"
	@echo "  download_results   pull the run's full output dir"
	@echo "                       [RUN_ID=...] [DEST=<local dir>]"
	@echo "  download_csv       pull just the sweep results CSV(s) for the run"
	@echo "                       [RUN_ID=...] [DEST=<local dir>]"
	@echo
	@echo "Bucket:  s3://$(BUCKET)/$(PREFIX)/"
	@echo "Inputs:  s3://$(BUCKET)/$(PREFIX)/inputs/"
	@echo "Outputs: s3://$(BUCKET)/$(PREFIX)/outputs/runs/<RUN_ID>/"

# --------------------------------------------------------------------------- #

upload:
	@test -n "$(CONFIG)" || { echo "ERR: CONFIG=path/to/sweep.yaml required"; exit 1; }
	@test -n "$(S3_KEY)" || { echo "ERR: S3_KEY=sub/path/sweep.yaml required"; exit 1; }
	@test -f "$(CONFIG)" || { echo "ERR: $(CONFIG) does not exist"; exit 1; }
	aws s3 cp "$(CONFIG)" "$(INPUT_URI)"
	@echo "Uploaded -> $(INPUT_URI)"

submit:
	@test -n "$(S3_KEY)" || { echo "ERR: S3_KEY=sub/path/sweep.yaml required"; exit 1; }
	@if [ -n "$(CONFIG)" ]; then \
	  test -f "$(CONFIG)" || { echo "ERR: $(CONFIG) does not exist"; exit 1; }; \
	  echo "[1/2] Uploading $(CONFIG) -> $(INPUT_URI)"; \
	  aws s3 cp "$(CONFIG)" "$(INPUT_URI)"; \
	fi
	@RID=run-$$(date -u +%Y%m%dT%H%M%SZ)-$$(uuidgen | cut -c1-6); \
	echo "[2/2] Submit  run_id=$$RID  ref=$(REF)  config=$(INPUT_URI)"; \
	aws --region $(REGION) lambda invoke \
	  --function-name $(LAMBDA) \
	  --cli-binary-format raw-in-base64-out \
	  --payload "$$(jq -nc \
	    --arg jobdef "$(JOBDEF)" \
	    --arg ref    "$(REF)" \
	    --arg s3uri  "$(INPUT_URI)" \
	    --arg rid    "$$RID" \
	    '{job_definition:$$jobdef, training_ref:$$ref, sweep_config_s3_uri:$$s3uri, run_id:$$rid}')" \
	  .aero-last-resp.json >/dev/null; \
	jq . .aero-last-resp.json; \
	jq -r '.jobId' .aero-last-resp.json > .aero-last-job-id; \
	jq -r '.runId' .aero-last-resp.json > .aero-last-run-id; \
	echo "Outputs -> s3://$(BUCKET)/$(PREFIX)/outputs/runs/$$RID/"

status:
	@test -n "$(JOB_ID)" || { echo "ERR: no JOB_ID (pass JOB_ID=... or run 'make submit' first)"; exit 1; }
	@trap 'echo; echo "(stopped)"; exit 0' INT; \
	while :; do \
	  echo "JOB_ID:  $(JOB_ID)"; \
	  echo "RUN_ID:  $(RUN_ID)"; \
	  echo "Outputs: $(OUTPUTS_URI)/"; \
	  echo "(refreshes every 15s, Ctrl+C to stop)"; \
	  echo; \
	  aws --region $(REGION) batch describe-jobs --jobs $(JOB_ID) \
	    --query 'jobs[0].{status:status, reason:statusReason, started:startedAt, stopped:stoppedAt}' \
	    --output table; \
	  STATUS=$$(aws --region $(REGION) batch describe-jobs --jobs $(JOB_ID) \
	    --query 'jobs[0].status' --output text); \
	  case "$$STATUS" in \
	    SUCCEEDED|FAILED) echo; echo "Job finished: $$STATUS"; break;; \
	  esac; \
	  sleep 15; \
	done

logs:
	@test -n "$(JOB_ID)" || { echo "ERR: no JOB_ID"; exit 1; }
	@LOG=$$(aws --region $(REGION) batch describe-jobs --jobs $(JOB_ID) \
	  --query 'jobs[0].container.logStreamName' --output text); \
	if [ "$$LOG" = "None" ] || [ -z "$$LOG" ]; then \
	  echo "No log stream yet - job is still in SUBMITTED/PENDING/RUNNABLE/STARTING."; \
	  echo "(Cold start can take a few additional minutes.)"; exit 1; \
	fi; \
	echo "Tailing $$LOG (Ctrl+C to detach; job keeps running)"; \
	aws --region $(REGION) logs tail /aws/batch/job \
	  --log-stream-names "$$LOG" --follow

cancel:
	@test -n "$(JOB_ID)" || { echo "ERR: no JOB_ID (pass JOB_ID=... or run 'make submit' first)"; exit 1; }
	@aws --region $(REGION) batch terminate-job \
	  --job-id $(JOB_ID) \
	  --reason "Cancelled by user via make cancel"
	@echo "Cancellation requested for $(JOB_ID). Status will move to FAILED shortly."

# Pulls the full run output directory. Uses `aws s3 cp --recursive` which lists
# all keys under the prefix and downloads each - works fine despite S3 not
# having real directories.
download_results:
	@test -n "$(RUN_ID)" || { echo "ERR: no RUN_ID (pass RUN_ID=... or run 'make submit' first)"; exit 1; }
	@D=$${DEST:-./downloaded/$(RUN_ID)}; \
	mkdir -p "$$D"; \
	aws s3 cp --recursive "$(OUTPUTS_URI)/" "$$D/"; \
	echo "Downloaded -> $$D"

# Pulls just the sweep results CSV(s) - the file you actually want for
# analysis. The filename embeds a timestamp (results_<ts>.csv) so we can't
# predict the exact name, but we can match the pattern.
download_csv:
	@test -n "$(RUN_ID)" || { echo "ERR: no RUN_ID"; exit 1; }
	@D=$${DEST:-./downloaded/$(RUN_ID)}; \
	mkdir -p "$$D"; \
	aws s3 cp --recursive "$(OUTPUTS_URI)/logs/sweep/" "$$D/" \
	  --exclude '*' --include 'results_*.csv'; \
	echo "Downloaded CSV(s) -> $$D"
