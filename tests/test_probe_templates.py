from pyneuroscope.probe_templates import (
    generate_sequential_shank_groups,
    load_builtin_templates,
    models_for_vendor,
    vendors,
)


def test_loads_builtin_templates() -> None:
    templates = load_builtin_templates()

    assert "NeuroNexus" in vendors(templates)
    assert models_for_vendor(templates, "NeuroNexus")


def test_generates_sequential_multi_probe_shank_groups() -> None:
    groups = generate_sequential_shank_groups(
        n_channels=16,
        n_probes=2,
        shanks_per_probe=2,
        channels_per_shank=4,
    )

    assert [group.name for group in groups] == [
        "probe1_shank1",
        "probe1_shank2",
        "probe2_shank1",
        "probe2_shank2",
    ]
    assert groups[2].channels == [8, 9, 10, 11]
