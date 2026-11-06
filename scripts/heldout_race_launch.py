#!/usr/bin/env python3
"""HELD-OUT checkable races (11-03): scripts/race_launch.py with five more tasks, for concepts in
configs/tica/final_split_v5.json "heldout" whose answers can be checked. Same flags, same protocol
(refit only after a pass; --refit-every is refused), same node script and scoring job
(tica_task_race_report.py). This wrapper only registers the tasks, so race_launch.py itself is
untouched.

  pyeval    "Python code"       HumanEval-Python (openai_humaneval rev 7dce605, 164; unit tests)
  javaeval  "Java code"         HumanEval-Java (MultiPL-E humaneval-java rev 28441b6, 158;
                                `java -ea` asserts, cvmfs java/21.0.1 + javatuples-1.2)
  mgsm_fr / mgsm_ru / mgsm_th   MGSM (juletxara/mgsm rev b2f13d4, 250 each): French / Russian /
                                Thai language; correct = the number, plus fastText in-language

Inputs (build_task_race_data.py from final_v5_clean.json + final_v5_elicit_inputs.json):
mech_1025/taskrace/{data,inputs,gold}_<task>.json.

  python3 scripts/heldout_race_launch.py --spec configs/race/heldout_v4d.json [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_launch

HELDOUT_TASKS = {
    # task -> (topic, answer_max)
    "pyeval": ("Python code", 768),
    "javaeval": ("Java code", 768),
    "mgsm_fr": ("French language", 512),
    "mgsm_ru": ("Russian language", 512),
    "mgsm_th": ("Thai language", 512),
    # 11-06 graded additions (build_task_race_data.py / capability_heldout docstrings)
    "mbppjava": ("Java code", 768),
    "mgsm_de": ("German language", 512),
    "mgsm_ja": ("Japanese language", 512),
    "belebele_fi": ("Finnish language", 512),
    "belebele_pt": ("Portuguese language", 512),
    "belebele_tr": ("Turkish language", 512),
    "belebele_el": ("Greek language", 512),
    "belebele_ta": ("Tamil language", 512),
    "mmlu_physics": ("physics", 512),
    "mmlu_cs": ("computer science", 512),
    "mmlu_bio": ("biology", 512),
    "mmlu_med": ("medicine and hospitals", 512),
}


def register(tasks: dict = race_launch.TASKS, task_dir: str = race_launch.TASK_DIR) -> dict:
    for t, (topic, amax) in HELDOUT_TASKS.items():
        tasks.setdefault(t, (topic, f"{task_dir}/data_{t}.json", f"{task_dir}/inputs_{t}.json",
                             f"{task_dir}/gold_{t}.json", amax, "tica_task_race_report.py"))
    return tasks


if __name__ == "__main__":
    register()
    race_launch.main()
