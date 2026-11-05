# Replication analysis scripts (TamIA, 11-03 / 11-04)

Stdlib-only readers behind the REPLICATION.md entries of 11-03/11-04. Paths inside are TamIA absolute paths
(`/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/...`); rebase them when running elsewhere.

- `kcmpn.py` / `kcmp.py`: bin a run's train_log over global steps (resume-aware). `KK=key1,key2 python3 kcmpn.py name run bins`
  (kcmpn skips NaN, e.g. beatk_turns_k when probe k was never beaten in a step).
- `rtab.py`, `beatcmp.py`, `capcmp.py`, `tables.py`: race tables per probe k (beaten / faced, turns to beat, accuracy of passing answers, fluke share).
- `battery2.py`: eval battery at a checkpoint (val race, quality races, persistence, capability, passive hide-and-solve).
- `mathwrong.py`, `mathrep.py`: GSM8K race passing answers, correct vs kind of wrong; repeated lines.
- `codeform.py`, `lang.py`, `lang2.py`, `goldcheck.py`: answer form (Python / JS / braces), language of MGSM answers, gold checks.
- `cfan.py`: counterfactual-repair races (cf field): control-passing answers whose repaired version is caught, per probe depth. `python3 cfan.py RACEDIR js|py`
- `jsfail.py`, `repair.py`: rerun Node / Python tests on race answers with stderr; mechanical syntax repair then retest
  (needs `capability_heldout.py` from heldout-tasks and a node binary: run in a CPU job).
