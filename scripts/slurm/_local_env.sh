# Sourced by job scripts (REPLICATION 2026-10-13k / 14e). Picks the FASTEST working environment:
# 1. if PY is unset (or the home default) and the fast venv tarball exists, use the fast venv
#    (torch 2.9.1 + fla fwd/bwd + causal-conv1d; REPLICATION 10-13c);
# 2. if PY's venv has a sibling <venv>.tar, unpack it to node-local $SLURM_TMPDIR and run from there:
#    importing torch from Lustre /scratch stalled jobs 15-25+ min when the filesystem was busy.
FAST_VENV=/scratch/sneheel/venvs/q35t29
HOME_PY=/home/sneheel/chameleons-qwen35/.venv/bin/python
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
# Colocated vLLM generation (REPLICATION 2026-10-14s; tica/vllm_gen.py): TICA_VLLM=1 unpacks the
# vllm-serving venv node-local and exports its python for the server subprocess.
if [ "${TICA_VLLM:-0}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ] && [ -f /scratch/sneheel/venvs/vllm-serving.tar ]; then
  t0=$(date +%s)
  tar -xf /scratch/sneheel/venvs/vllm-serving.tar -C "$SLURM_TMPDIR"
  export TICA_VLLM_PY="$SLURM_TMPDIR/vllm-serving/bin/python"
  echo "vllm venv: $TICA_VLLM_PY ($(( $(date +%s) - t0 ))s)"
fi
# Persistent Triton kernel + autotuning cache (REPLICATION 2026-10-15d): a per-job $SLURM_TMPDIR
# cache made EVERY job recompile / re-autotune the fla kernels (bench 5656340: the same read workload
# took 944 s cold vs 270 s warm; 330 s vs 12 s for the first call). A shared cache DIRECTORY on
# /scratch hit the account's 1M-file quota within hours (86k files; 5667301 failed), so the cache is
# kept as ONE tarball per venv: unpacked node-local at job start, written back at job exit (tmp file
# + atomic rename, only if the local cache grew). TICA_TRITON_LOCAL=1 = the old empty per-job cache.
export TRITON_CACHE_AUTOTUNING=1
if [ "${TICA_TRITON_LOCAL:-0}" != "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  _TT=/scratch/sneheel/triton_cache_$(basename "$(dirname "$(dirname "$PY")")").tar
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
