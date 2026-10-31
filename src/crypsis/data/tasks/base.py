"""Checkable tasks: the `Task` record, the `TASKS` mapping concept -> Task, and the `check_task`
dispatcher. Functional where possible (SQL: execution match against the reference tables; JSON: the
requested function schema or a ToolACE spec; JavaScript: APPS tests under Node), else the FORM check
(parses / right markers / right script). Ported from tica/validity.py; behaviour is identical.

Resources a check may need (the fastText model, the APPS tasks, the node binary) travel in a
`TaskEnv` argument, never in module state or the environment."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from crypsis.data.tasks import apps_js
from crypsis.data.tasks.form import (
    FORM_CHECKS,
    LANG_CODES,
    LanguageID,
    LIDUnavailable,
)
from crypsis.data.tasks.json_spec import json_schema_ok, json_spec_ok
from crypsis.data.tasks.sql import sql_ref_match, sql_split


@dataclass
class TaskEnv:
    """Resources for checks: language ID model, APPS tasks by "apps:<id>" ref, node binary."""

    lid: LanguageID | None = None
    apps: Mapping[str, dict] = field(default_factory=dict)
    node: str | None = None
    timeout: float = 10.0


@dataclass(frozen=True)
class Task:
    """One checkable concept. `form(answer, env)` is the shape check; `check(answer, request, ref,
    env)` is the functional check where one exists (else the same as `form`)."""

    concept: str
    family: str  # "language" | "format"
    form: Callable[[str, TaskEnv], bool]
    check: Callable[[str, str, str | None, TaskEnv], bool]
    kind: str = "form"  # "form" | "functional" (some items)


def _form_of(concept: str) -> Callable[[str, TaskEnv], bool]:
    fn = FORM_CHECKS[concept]
    return lambda x, env: fn(x)


def _lang_form(concept: str) -> Callable[[str, TaskEnv], bool]:
    def run(x: str, env: TaskEnv) -> bool:
        if env.lid is None:
            raise LIDUnavailable("language ID needs TaskEnv(lid=LanguageID(model_path))")
        return env.lid.is_language(x, concept)

    return run


def _sql_check(answer: str, request: str, ref: str | None, env: TaskEnv) -> bool:
    if ref is not None:
        m = sql_ref_match(answer, ref)
        if m is not None:
            return m
    return _form_of("SQL code")(answer, env)


def _json_check(answer: str, request: str, ref: str | None, env: TaskEnv) -> bool:
    if ref and str(ref).startswith("jsonspec:"):
        return json_spec_ok(answer, json.loads(str(ref)[len("jsonspec:"):]))
    return json_schema_ok(answer, request, ref)


def _js_check(answer: str, request: str, ref: str | None, env: TaskEnv) -> bool:
    if ref and str(ref).startswith("apps:"):
        it = env.apps.get(ref)
        return it is not None and apps_js.check_item(it, answer, env.node, env.timeout)
    return _form_of("JavaScript code")(answer, env)


def _build() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for c in FORM_CHECKS:
        form = _form_of(c)
        tasks[c] = Task(c, "format", form, lambda a, q, r, e, f=form: f(a, e))
    tasks["SQL code"] = Task("SQL code", "format", _form_of("SQL code"), _sql_check, "functional")
    tasks["JSON data"] = Task("JSON data", "format", _form_of("JSON data"), _json_check,
                              "functional")
    tasks["JavaScript code"] = Task("JavaScript code", "format", _form_of("JavaScript code"),
                                    _js_check, "functional")
    for c in LANG_CODES:
        form = _lang_form(c)
        tasks[c] = Task(c, "language", form, lambda a, q, r, e, f=form: f(a, e))
    return tasks


TASKS: dict[str, Task] = _build()
FAMILY = {c: t.family for c, t in TASKS.items()}


def checkable(concept: str) -> bool:
    """Is there a deterministic check for this concept?"""
    return concept in TASKS


def check(concept: str, answer: str, env: TaskEnv | None = None) -> bool:
    """FORM check of a checkable concept; never raises on model text (except LIDUnavailable)."""
    try:
        return bool(TASKS[concept].form(answer or "", env or TaskEnv()))
    except LIDUnavailable:
        raise
    except Exception:
        return False


def functional(concept: str, ref: str | None = None) -> bool:
    """Is there a functional check for this item (SQL needs a runnable reference)?"""
    return concept == "JSON data" or (concept == "SQL code" and ref is not None
                                      and sql_split(ref) is not None)


def check_task(concept: str, answer: str, request: str = "", ref: str | None = None, *,
               env: TaskEnv | None = None) -> bool:
    """Does the answer DO the task? Functional where possible, else the form check. Gates the
    trainer's cap_free NLL. Never raises on model text; a missing language-ID model raises
    LIDUnavailable (it never silently fails every answer)."""
    try:
        return bool(TASKS[concept].check(answer or "", request, ref, env or TaskEnv()))
    except LIDUnavailable:
        raise
    except Exception:
        return False
