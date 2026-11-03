# Sourced by job scripts (REPLICATION 2026-10-13k / 14e). Picks the FASTEST working environment:
# 1. if PY is unset (or the home default) and the fast venv tarball exists, use the fast venv
#    (torch 2.9.1 + fla fwd/bwd + causal-conv1d; REPLICATION 10-13c);
# 2. if PY's venv has a sibling <venv>.tar, unpack it to node-local $SLURM_TMPDIR and run from there:
#    importing torch from Lustre /scratch stalled jobs 15-25+ min when the filesystem was busy.
FAST_VENV=/scratch/s/sneheel/venvs/q35t29
HOME_PY=/scratch/s/sneheel/venvs/q35t29/bin/python
if { [ -z "${PY:-}" ] || [ "$PY" = "$HOME_PY" ]; } && [ -f "$FAST_VENV.tar" ]; then
  PY=$FAST_VENV/bin/python
fi
PY=${PY:-$HOME_PY}
if [ -n "${SLURM_TMPDIR:-}" ]; then
  VENV=$(dirname "$(dirname "$PY")")
  if [ -f "$VENV.tar" ]; then
    t0=$(date +%s)
    tar -xf "$VENV.tar" -C "$SLURM_TMPDIR"
    PY="$SLURM_TMPDIR/$(basename "$VENV")/bin/python"
    echo "local venv: $PY ($(( $(date +%s) - t0 ))s)"
  fi
fi
# Colocated vLLM generation (REPLICATION 2026-10-14s; tica/vllm_gen.py). TICA_VLLM=1: provide the
# vLLM server python unless TICA_VLLM_PY is set (11-03; the old vllm-serving.tar was Narval-only).
#   TICA_VLLM_VENV  venv name under /scratch/s/sneheel/venvs: vllm-latest (default; vLLM 0.21, Qwen3.5
#                   incl. the merged-weight reload) or vllm-g4 (vLLM 0.25; needed for Gemma-4).
# Its <venv>.tar is unpacked into $SLURM_TMPDIR when present (imports from Lustre stall under load;
# skipped if already unpacked there), else the venv directory is used in place. Also exports
# TICA_VLLM_MODULES = the module set that venv was built with (unless set). A job that already set
# TICA_VLLM_PY (e.g. a node script that unpacked the venv once for all its slots) is left alone.
if [ "${TICA_VLLM:-0}" = "1" ]; then
  _VV=${TICA_VLLM_VENV:-vllm-latest}
  _VD=/scratch/s/sneheel/venvs/$_VV
  case "$_VV" in
    vllm-g4) _VM="StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv" ;;
    *) _VM="StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6" ;;
  esac
  export TICA_VLLM_MODULES=${TICA_VLLM_MODULES:-$_VM}
  if [ -z "${TICA_VLLM_PY:-}" ]; then
    t0=$(date +%s)
    if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$_VD.tar" ]; then
      [ -x "$SLURM_TMPDIR/$_VV/bin/python" ] || tar -xf "$_VD.tar" -C "$SLURM_TMPDIR"
      export TICA_VLLM_PY="$SLURM_TMPDIR/$_VV/bin/python"
    elif [ -x "$_VD/bin/python" ]; then
      export TICA_VLLM_PY="$_VD/bin/python"
    else
      echo "TICA_VLLM=1: no vLLM venv at $_VD" >&2
    fi
    echo "vllm venv: ${TICA_VLLM_PY:-none} ($(( $(date +%s) - t0 ))s; modules: $TICA_VLLM_MODULES)"
  fi
fi
# Persistent Triton kernel + autotuning cache (REPLICATION 2026-10-15d): a per-job $SLURM_TMPDIR
# cache made EVERY job recompile / re-autotune the fla kernels (bench 5656340: the same read workload
# took 944 s cold vs 270 s warm; 330 s vs 12 s for the first call). A shared cache DIRECTORY on
# /scratch hit the account's 1M-file quota within hours (86k files; 5667301 failed), so the cache is
# kept as ONE tarball per venv: unpacked node-local at job start, written back at job exit (tmp file
# + atomic rename, only if the local cache grew). TICA_TRITON_LOCAL=1 = the old empty per-job cache.
export TRITON_CACHE_AUTOTUNING=1
if [ "${TICA_TRITON_LOCAL:-0}" != "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  _TT=/scratch/s/sneheel/triton_cache_$(basename "$(dirname "$(dirname "$PY")")").tar
  export TRITON_CACHE_DIR=$SLURM_TMPDIR/triton
  mkdir -p "$TRITON_CACHE_DIR"
  [ -f "$_TT" ] && tar -xf "$_TT" -C "$TRITON_CACHE_DIR" 2>/dev/null
  _TT_N0=$(find "$TRITON_CACHE_DIR" -type f | wc -l)
  _tt_save() {
    local n; n=$(find "$TRITON_CACHE_DIR" -type f | wc -l)
    if [ "$n" -gt "$_TT_N0" ]; then
      tar -cf "$_TT.tmp.$$" -C "$TRITON_CACHE_DIR" . 2>/dev/null && mv -f "$_TT.tmp.$$" "$_TT"
    fi
  }
  trap _tt_save EXIT
fi
