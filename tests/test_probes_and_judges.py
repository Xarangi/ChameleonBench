from next_chameleons.activations import SyntheticActivationExtractor
from next_chameleons.datasets import SyntheticDatasetAdapter
from next_chameleons.judges import EnsembleJudge
from next_chameleons.probes.attention import AttentionProbe
from next_chameleons.probes.geometry import GeometryProbe
from next_chameleons.probes.linear import LinearProbe
from next_chameleons.probes.mlp import MLPProbe
from next_chameleons.selectors import ActiveProbePoolSelector


def _batch():
    bundle = SyntheticDatasetAdapter(num_examples=80).load()
    return SyntheticActivationExtractor(
        hidden_dim=12,
        num_layers=3,
        concept_shift=1.6,
    ).extract(bundle)


def test_probe_reports_detect_synthetic_concept() -> None:
    batch = _batch()
    probe = LinearProbe().fit(batch)
    report = probe.report(batch)

    assert report.accuracy >= 0.8
    assert report.auc >= 0.8


def test_ensemble_judge_aggregates_probe_reports() -> None:
    batch = _batch()
    judge = EnsembleJudge(
        judge_id="test",
        probes=[LinearProbe(), MLPProbe(), AttentionProbe(), GeometryProbe()],
        aggregation="max",
        threshold=0.25,
    ).fit(batch)

    result = judge.evaluate(batch)

    assert result.flagged is True
    assert len(result.probe_reports) == 4


def test_active_probe_selector_promotes_candidates() -> None:
    batch = _batch()
    result = ActiveProbePoolSelector(promote_top_k=2).select(batch)

    assert len(result.promoted) == 2
    assert len(result.scores) == 2
    assert result.scores[0].score >= result.scores[1].score
