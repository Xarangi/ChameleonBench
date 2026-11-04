# Moving training to Narval (11-04)

What is in the TamIA bundle (`/project/aip-rrabba/sneheel/narval_bundle/`, built by `make_bundle.sh` there).
It holds no model weights, adapters or checkpoints. Paths inside the tars are relative to `/scratch/s/sneheel`.

| File | Contents | Needed for |
|---|---|---|
| `chameleon_data_core.tar.zst` (496 MB) | `next_chameleons_raw_cache/` (final_v5 data + elicit inputs, 27B concept labels, Dolly, capability pools: APPS-JS, gretel SQL, ToolACE JSON, GSM8K, MBPP, HumanEval-Py/JS/Java, MGSM, MMLU, raw downloads, fastText lid.176); 4B base probe bank `l1_base/qwen35_4b/probes`; `answer_probe_cache`; anchor caches (4B, 2B); GSM8K base solutions (`mathrl_4b_full`, `mathrl_qwen35_2b_full` base_gens); race inputs (`mech_1025/taskrace/{data,inputs,gold}_*.json`, `mech_1025/mathrace` GSM8K test + gold) | training any v4d-family 2B/4B variant; races |
| `chameleon_panel_probe_cache.tar.zst` (73 GB raw) | base-model turn-0 panel probes, content-hashed | optional: saves GPU; recomputable with `scripts/slurm/tamia_panel_precompute.sh` |
| `requirements_q35t29.txt`, `requirements_vllm-latest.txt` | exact package versions of the training venv and the vLLM race venv | rebuilding the venvs |
| `SHA256SUMS` | checksums | |

## Commands (run on a Narval login node)

```bash
# 1. copy (asks for TamIA password + MFA); skip the panel cache with --exclude if not wanted
rsync -avP --exclude 'chameleon_panel_probe_cache.tar.zst' \
  sneheel@tamia.alliancecan.ca:/project/aip-rrabba/sneheel/narval_bundle/ $SCRATCH/narval_bundle/
cd $SCRATCH/narval_bundle && sha256sum -c --ignore-missing SHA256SUMS

# 2. unpack under $SCRATCH (same layout as /scratch/s/sneheel on TamIA)
cd $SCRATCH && zstd -dc narval_bundle/chameleon_data_core.tar.zst | tar -xf -
# optional: zstd -dc narval_bundle/chameleon_panel_probe_cache.tar.zst | tar -xf -

# 3. code
git clone -b tamia https://github.com/Xarangi/ChameleonBench.git $SCRATCH/ChameleonBench
cd $SCRATCH/ChameleonBench && git switch -c narval
grep -rlI /scratch/s/sneheel configs scripts src | xargs sed -i "s#/scratch/s/sneheel#$SCRATCH#g"
grep -rlI aip-rrabba configs scripts | xargs sed -i "s#aip-rrabba#<narval account, e.g. def-rrabba>#g"

# 4. base models (compute nodes have no internet; download on the login node). Same revisions as TamIA:
export HF_HOME=$SCRATCH/hf
huggingface-cli download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
huggingface-cli download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
```

Venvs: configs and scripts expect `$SCRATCH/venvs/q35t29` (training, Python 3.12) and `$SCRATCH/venvs/vllm-latest`
(races, Python 3.11, torch 2.11 / vLLM 0.21 Compute Canada wheels). Build them from the requirements files with the
Alliance wheelhouse (`module load python/3.12; virtualenv --no-download ...; pip install --no-index -r ...`); drop pins
that the wheelhouse lacks.

## Differences that matter

- GPUs: Narval has 4x A100 **40 GB** per node; TamIA has H100 80 GB. The 4B configs use read batch 64 (~59-67 GB
  peak of 80) and races run vLLM at 0.35 + HF at 0.60 of one GPU. Expect to halve `model.batch_size` (reads are
  exact under re-batching) and to check the rollout / race memory before long runs.
- Slurm account: `aip-rrabba` is the TamIA (PAICE) allocation; use the Narval one.
- `eval_on_save` (`scripts/slurm/tica_val.sh`) requests A100 gres by default; that works on Narval, unlike TamIA.
- No 9B base probe bank exists (`l1_base/qwen35_9b` is referenced by scripts but was never built).
- Each run writes its 25% capability hold-out to `<run>/adaptive/cap_free_holdout.json`; new runs make their own.
