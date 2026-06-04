#!/usr/bin/env bash
# Stage selected Hugging Face model and dataset cache directories from shared
# scratch to node-local storage. This avoids repeated random/metadata-heavy
# reads from Lustre while keeping durable caches and outputs in $SCRATCH.

stage_next_chameleons_hf_cache() {
  local repo_root="${REPO_ROOT:-$(pwd)}"
  local persistent_hf_home="${NEXT_CHAMELEONS_PERSISTENT_HF_HOME:-${SCRATCH:-${repo_root}}/.cache/huggingface}"
  local stage_enabled="${NEXT_CHAMELEONS_STAGE_HF_CACHE:-1}"
  local stage_models="${NEXT_CHAMELEONS_STAGE_MODELS:-}"
  local stage_datasets="${NEXT_CHAMELEONS_STAGE_DATASETS:-}"

  export NEXT_CHAMELEONS_PERSISTENT_HF_HOME="${persistent_hf_home}"

  if [[ "${stage_enabled}" != "1" || -z "${SLURM_TMPDIR:-}" ]]; then
    export HF_HOME="${persistent_hf_home}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    return 0
  fi

  if [[ -z "${stage_models}" && -z "${stage_datasets}" ]]; then
    export HF_HOME="${persistent_hf_home}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
    return 0
  fi

  local local_hf_home="${SLURM_TMPDIR}/next_chameleons_hf"
  local local_hub="${local_hf_home}/hub"
  mkdir -p "${local_hub}" "${local_hf_home}/datasets" "${local_hf_home}/transformers" "${SLURM_TMPDIR}/tmp"

  echo "Staging HF cache to node-local storage:"
  echo "  persistent=${persistent_hf_home}"
  echo "  local=${local_hub}"
  echo "  models=${stage_models}"
  echo "  datasets=${stage_datasets}"
  df -h "${SLURM_TMPDIR}" || true

  local model dataset cache_name source_dir target_dir
  for model in $(printf '%s' "${stage_models}" | tr ',;' '  '); do
    [[ -z "${model}" ]] && continue
    cache_name="models--${model//\//--}"
    source_dir="${persistent_hf_home}/${cache_name}"
    if [[ ! -d "${source_dir}" && -d "${persistent_hf_home}/hub/${cache_name}" ]]; then
      source_dir="${persistent_hf_home}/hub/${cache_name}"
    fi
    if [[ ! -d "${source_dir}" ]]; then
      echo "Missing persistent HF cache for ${model}: expected ${persistent_hf_home}/${cache_name}" >&2
      exit 4
    fi
    target_dir="${local_hub}/${cache_name}"
    mkdir -p "${target_dir}"
    echo "  rsync ${model}"
    rsync -a --delete --human-readable --info=stats2 "${source_dir}/" "${target_dir}/"
  done

  for dataset in $(printf '%s' "${stage_datasets}" | tr ',;' '  '); do
    [[ -z "${dataset}" ]] && continue
    cache_name="datasets--${dataset//\//--}"
    source_dir="${persistent_hf_home}/${cache_name}"
    if [[ ! -d "${source_dir}" && -d "${persistent_hf_home}/hub/${cache_name}" ]]; then
      source_dir="${persistent_hf_home}/hub/${cache_name}"
    fi
    if [[ ! -d "${source_dir}" ]]; then
      echo "Missing persistent HF dataset cache for ${dataset}: expected ${persistent_hf_home}/${cache_name}" >&2
      exit 5
    fi
    target_dir="${local_hub}/${cache_name}"
    mkdir -p "${target_dir}"
    echo "  rsync dataset ${dataset}"
    rsync -a --delete --human-readable --info=stats2 "${source_dir}/" "${target_dir}/"
  done

  export HF_HOME="${local_hf_home}"
  export HF_HUB_CACHE="${local_hub}"
  export HF_DATASETS_CACHE="${local_hf_home}/datasets"
  export TRANSFORMERS_CACHE="${local_hf_home}/transformers"
  export TMPDIR="${SLURM_TMPDIR}/tmp"
  export NEXT_CHAMELEONS_LOCAL_HF_STAGED=1

  echo "Using node-local HF cache:"
  echo "  HF_HOME=${HF_HOME}"
  echo "  HF_HUB_CACHE=${HF_HUB_CACHE}"
  echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  du -sh "${local_hub}"/models--* "${local_hub}"/datasets--* 2>/dev/null || true
}
