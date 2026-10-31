"""Capability pools for the free-generation term: per checkable concept, the concept's own
calib / episode items plus external task sources (APPS JavaScript, gretel SQL, ToolACE JSON, Dolly),
and a deterministic hold-out split (by request hash) so evals use items training never saw.
Ported from tica/free_pools.py and scripts/tica.py::cap_free_requests; same rows in the same order.
Pools import the tasks, never the reverse. Aggregates only: generated answers are never stored."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Collection, Mapping
from pathlib import Path

from crypsis.data.concepts import ConceptSet, text_id
from crypsis.data.tasks.apps_js import AppsJsTasks
from crypsis.data.tasks.base import FAMILY, TaskEnv, check_task, checkable
from crypsis.data.tasks.json_spec import (  # noqa: F401  (re-exported pool API)
    json_spec_ok,
    json_spec_request,
    materialize_toolace_json,
    toolace_json_rows,
)
from crypsis.data.tasks.sql import (  # noqa: F401  (re-exported pool API)
    gretel_sql_rows,
    materialize_gretel_sql,
    sql_split,
    sql_task_request,
)

MAX_CONTEXT_CHARS = 1200  # Dolly items with a long context are dropped (prompt length)
Row = tuple[str, str | None, str | None]  # (request, fallback answer, reference)


def directive(requests: list[str]) -> str | None:
    """The concept's answer directive: the most common last line of its existing requests (None if
    no line is shared by at least a quarter of them)."""
    last = [r.strip().splitlines()[-1].strip() for r in requests if r.strip()]
    if not last:
        return None
    line, n = Counter(last).most_common(1)[0]
    return line if n >= max(2, len(last) // 4) else None


def dolly_rows(path: str | Path, concepts: dict[str, str], n_per_concept: int,
               seed: int = 0) -> dict[str, list[tuple[str, None, None]]]:
    """{concept: [(request, None, None)]} from Dolly-15k (databricks-dolly-15k, CC-BY-SA-3.0):
    disjoint shuffled slices of n_per_concept items, each request = instruction (+ context) + the
    concept's directive (concepts: {name: directive}). No fallback answer."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ctx = (r.get("context") or "").strip()
            if len(ctx) > MAX_CONTEXT_CHARS or not r.get("instruction", "").strip():
                continue
            body = r["instruction"].strip() + (f"\n\n{ctx}" if ctx else "")
            items.append(body)
    random.Random(seed).shuffle(items)
    out, start = {}, 0
    for c in sorted(concepts):
        chunk = items[start : start + n_per_concept]
        start += n_per_concept
        out[c] = [(f"{b}\n\n{concepts[c]}", None, None) for b in chunk]
    return out


def _held(request: str, frac: float) -> bool:
    h = int(hashlib.sha256(request.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def split_holdout(pool: dict[str, list], frac: float) -> tuple[dict[str, list], dict[str, list]]:
    """(training pool, held-out pool): a deterministic `frac` of every concept's rows (by request
    hash) is held out. frac <= 0 -> (pool, {})."""
    if frac <= 0:
        return pool, {}
    train, held = {}, {}
    for c, rows in pool.items():
        held[c] = [r for r in rows if _held(r[0], frac)]
        train[c] = [r for r in rows if not _held(r[0], frac)]
    return train, held


def build_free_pool(concepts: ConceptSet, requests: Mapping[str, str],
                    concept_names: Collection[str], *, apps_js: str | Path | None = None,
                    gretel_sql: str | Path | None = None, toolace_json: str | Path | None = None,
                    dolly: str | Path | None = None, dolly_n: int = 1500,
                    js_keep_form: bool = False, env: TaskEnv | None = None
                    ) -> dict[str, list[Row]]:
    """{checkable train concept: [(request, fallback answer, reference)]}.

    Own items come from the calib and episode splits (training rollouts force fit-split texts: these
    are new). SQL items become execution-checkable (the request carries the reference's tables +
    rows, the fallback answer is the reference query; items whose reference does not run in sqlite
    are dropped). Other concepts: the published answer is both fallback and reference, kept only if
    it passes its own check. `requests` maps text_id -> request (RequestMap or dict). The optional
    sources are paths of materialized files: apps_js (APPS tasks, JavaScript concept; own rows kept
    after them when js_keep_form), gretel_sql (SQL), toolace_json (JSON), dolly (Dolly-15k jsonl,
    dolly_n per language / all-caps concept, each with the concept's own directive). `env` carries
    the language-ID model for the language concepts' own-item check."""
    names = set(concept_names)
    pool: dict[str, list[Row]] = {}
    for c in sorted(x for x in names if checkable(x)):
        rows: list[Row] = []
        for sp in ("calib", "episode"):
            for ct in concepts.concepts(sp, ("train",), only={c}):
                for x in ct.texts:
                    q = requests.get(text_id(x))
                    if not q:
                        continue
                    if c == "SQL code":
                        q2 = sql_task_request(q, x)
                        if q2 is not None:
                            rows.append((q2, sql_split(x)[1], x))
                    elif check_task(c, x, q, x, env=env):  # a reference failing its own check
                        rows.append((q, x, x))             # is no NLL target
        if rows:
            pool[c] = rows
    if apps_js and "JavaScript code" in names:
        apps = AppsJsTasks.load(apps_js).rows()
        pool["JavaScript code"] = apps + (pool.get("JavaScript code", []) if js_keep_form else [])
    if gretel_sql and "SQL code" in names:
        own = pool.get("SQL code", [])
        pool["SQL code"] = own + gretel_sql_rows(gretel_sql, directive([q for q, _, _ in own]))
    if toolace_json and "JSON data" in names:
        own = pool.get("JSON data", [])
        pool["JSON data"] = own + toolace_json_rows(toolace_json,
                                                    directive([q for q, _, _ in own]))
    if dolly:
        form = {}
        for c in pool:
            if FAMILY.get(c) == "language" or c == "all-caps text":
                d = directive([r[0] for r in pool[c]])
                if d:
                    form[c] = d
        for c, rows in dolly_rows(dolly, form, dolly_n).items():
            pool[c] = pool[c] + rows
    return pool
