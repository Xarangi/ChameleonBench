"""Guard against drift between configs/ and the packaged builtin_configs/."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
BUILTIN = REPO_ROOT / "src" / "next_chameleons" / "builtin_configs"


def _relative_yaml_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.yaml"))
    }


def test_builtin_configs_match_project_configs() -> None:
    """configs/ is the source of truth; the packaged copy must be identical.

    On failure run `scripts/sync_builtin_configs.sh`.
    """

    project = _relative_yaml_files(CONFIGS)
    builtin = _relative_yaml_files(BUILTIN)

    missing = sorted(set(project) - set(builtin))
    extra = sorted(set(builtin) - set(project))
    assert not missing, f"builtin_configs missing files (run sync script): {missing}"
    assert not extra, f"builtin_configs has extra files (run sync script): {extra}"
    different = sorted(
        name for name in project if project[name] != builtin[name]
    )
    assert not different, f"builtin_configs out of sync (run sync script): {different}"
