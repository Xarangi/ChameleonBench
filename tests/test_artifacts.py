from next_chameleons.artifacts import DatasetManifest, SafetyRedactor, SourceReference


def test_redactor_removes_raw_text_fields() -> None:
    payload = {
        "prompt": "secret raw prompt",
        "nested": {"completion_text": "secret completion", "metric": 0.8},
    }

    redacted = SafetyRedactor().redact(payload)

    assert redacted["prompt"] == "[REDACTED_RAW_TEXT]"
    assert redacted["nested"]["completion_text"] == "[REDACTED_RAW_TEXT]"
    assert redacted["nested"]["metric"] == 0.8


def test_manifest_requires_pinned_real_sources() -> None:
    manifest = DatasetManifest(
        dataset_id="paper_sources",
        source=SourceReference(
            kind="huggingface_dataset",
            name="example",
            revision="to-pin-before-real-run",
            checksum="to-record-after-download",
        ),
        split="train",
        num_examples=1,
        label_names=["no", "yes"],
    )

    try:
        manifest.validate()
    except ValueError as exc:
        assert "lacks pinned" in str(exc)
    else:
        raise AssertionError("Expected unpinned manifest to fail")
