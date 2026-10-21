#!/usr/bin/env bash
# =============================================================================
# Strata on Radeon 780M (gfx1103) — llmhost2, benchmark run
# =============================================================================
# What the engine gets and why (docs/RADEON_780M.md, vaskes/Strata @ 0.2.1):
#
#   STRATA_EXPERIMENTAL_GFX1103=1  REQUIRED. Without it setup.py drops gfx1103 from AMD_ARCHS,
#                                  sees the 780M as an unsupported card and refuses to write
#                                  the run settings for any family (verified: launch exit 5).
#   §4 exact speed switches (17)   setup writes these into the config's "env" by itself once the
#                                  opt-in is on — no need to pass them here.
#   §5 matrix-core prompt set       the engine turns this on by ITSELF when it detects an
#                                  integrated gfx1103 on Linux (STRATA_IGPU_PROMPT_SET).
#   §3 hipBLASLt table              loaded by the engine from the component package's tuning/
#                                  folder, matched via BUILD.json's hipblaslt_version (100401).
#                                  There is NO --hipblaslt-tuning flag: passing one makes the
#                                  engine print its help and exit (verified).
#   §6 --spec 2 is the best MTP window on this chip (15.65 tok/s vs 14.18 at 4 and 11.16 at 6);
#      --lookup-chain 4 adds +2.3%. setup's own --spec-min-p 0.5 measured 4.5% slower here.
#
# Memory: UD-IQ4_XS keeps 59.5 GiB of experts; on a UMA APU setup.py's unsloth_budget()
# (#1715) forces --resident-budget-gib and the rest is read from the SSD.
# =============================================================================

set -uo pipefail

IMG=strata:0.2.1-rocm-gfx1103
NAME=strata-780m
MODEL_DIR=/opt/models/Qwen3.8-Flash-Next-GGUF
CTX="${CTX:-32k}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-vaskes-strata-780m}"

# engine options + STRATA_* vars, passed as repeated `launch --set KEY=VALUE`
STRATA_SET_VALUE="spec=2 lookup-chain=4 spec-min-p=0.35 vram-reserve-mib=${VRAMRES:-6144} STRATA_HIPBLASLT_TABLE=${TBLMODE:-off} STRATA_IGPU_PROMPT_SET=${IGPUSET:-1}"

log() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

log "removing any old container"
docker rm -f "$NAME" 2>/dev/null || true

log "GPU + RAM state"
rocm-smi --showproductname 2>/dev/null | grep -i "card model" || true
free -g | head -2

log "starting $NAME (ctx=$CTX)"
docker run -d --name "$NAME" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --shm-size 4g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "127.0.0.1:${PORT}:8080" \
  -v strata-780m-data:/data \
  -v "${MODEL_DIR}:/models:ro" \
  -e MODEL=UD-IQ4_XS \
  -e MODEL_PATH=/models \
  -e "CONTEXT=${CTX}" \
  -e VISION=off \
  -e TUNE=skip \
  -e "API_KEY=${API_KEY}" \
  -e STRATA_EXPERIMENTAL_GFX1103=1 \
  -e "STRATA_SET=${STRATA_SET_VALUE}" \
  -e "STRATA_IGPU_PROMPT_SET=${IGPUSET:-1}" \
  -e "STRATA_HIPBLASLT_TABLE=${TBLMODE:-off}" \
  -e "STRATA_HIP_WMMA=${WMMA:-1}" \
  -e "STRATA_PF_GEMM=${WMMA:-1}" \
  -e "STRATA_PA_FAST=${WMMA:-1}" \
  -e "STRATA_SELECT_WMMA=${WMMA:-1}" \
  -e "STRATA_HC_UPMIX=${WMMA:-1}" \
  --restart unless-stopped \
  "$IMG"

log "container started"
sleep 25
docker ps --filter "name=$NAME" --format "{{.Names}} {{.Status}}"
echo
echo "boot log:   docker logs -f $NAME"
echo "health:     curl -s http://127.0.0.1:${PORT}/health -H 'Authorization: Bearer ${API_KEY}'"