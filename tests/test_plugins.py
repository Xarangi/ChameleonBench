from next_chameleons import plugins


class _FakeEntryPoint:
    name = "fake_plugin"

    def __init__(self) -> None:
        self.loaded = False

    def load(self):
        self.loaded = True
        return object()


class _FakeEntryPoints:
    def __init__(self, entry_point: _FakeEntryPoint) -> None:
        self.entry_point = entry_point

    def select(self, *, group: str):
        if group == "next_chameleons.plugins":
            return [self.entry_point]
        return []


def test_load_entrypoint_plugins_loads_external_registrars(monkeypatch) -> None:
    entry_point = _FakeEntryPoint()
    monkeypatch.setattr(plugins, "entry_points", lambda: _FakeEntryPoints(entry_point))

    loaded = plugins.load_entrypoint_plugins()

    assert loaded == ["fake_plugin"]
    assert entry_point.loaded is True
