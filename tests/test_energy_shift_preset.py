import numpy as np

from NepTrainKit.core.energy_shift import EnergyBaselinePreset, apply_energy_baseline
from NepTrainKit.core.structure import Structure


def _make_structure(config_type: str, species: list[str], energy: float) -> Structure:
    lattice = np.eye(3, dtype=np.float32)
    atomic_properties = {
        "species": np.asarray(species, dtype=object),
        "pos": np.zeros((len(species), 3), dtype=np.float32),
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    additional_fields = {
        "Config_type": config_type,
        "energy": float(energy),
    }
    return Structure(lattice, atomic_properties, properties, additional_fields)


def test_apply_energy_baseline_pattern_fallback_matches_new_config_types():
    structure = _make_structure("A/1", ["H", "O"], 10.0)
    preset = EnergyBaselinePreset(
        alignment_mode="REF_GROUP",
        elements=["H", "O"],
        group_to_ref={"A.*": [1.0, 2.0]},
        group_patterns=["A.*"],
        config_to_group={"A": "A.*"},
    )

    stats = apply_energy_baseline([structure], preset)

    assert structure.energy == 7.0
    assert stats["shifted_structures"] == 1
    assert stats["used_pattern_fallback"] == 1
    assert stats["unmatched_config_types"] == []


def test_apply_energy_baseline_reports_unmatched_config_types():
    structure = _make_structure("A/1", ["H", "O"], 10.0)
    preset = EnergyBaselinePreset(
        alignment_mode="REF_GROUP",
        elements=["H", "O"],
        group_to_ref={"B.*": [1.0, 2.0]},
        group_patterns=["B.*"],
        config_to_group={},
    )

    stats = apply_energy_baseline([structure], preset)

    assert structure.energy == 10.0
    assert stats["shifted_structures"] == 0
    assert stats["unmatched_config_types"] == ["A/1"]


def test_apply_energy_baseline_skips_structures_without_energy():
    structure = _make_structure("A/1", ["H", "O"], 10.0)
    structure.additional_fields.pop("energy", None)
    preset = EnergyBaselinePreset(
        alignment_mode="REF_GROUP",
        elements=["H", "O"],
        group_to_ref={"A.*": [1.0, 2.0]},
        group_patterns=["A.*"],
        config_to_group={},
    )

    stats = apply_energy_baseline([structure], preset)

    assert stats["shifted_structures"] == 0
    assert stats["skipped_no_energy"] == 1
    assert stats["unmatched_config_types"] == []


def test_apply_energy_baseline_writes_energy_original_once():
    structure = _make_structure("A/1", ["H", "O"], 10.0)
    preset_1 = EnergyBaselinePreset(
        alignment_mode="REF_GROUP",
        elements=["H", "O"],
        group_to_ref={"A/1": [1.0, 2.0]},
        group_patterns=[],
        config_to_group={"A/1": "A/1"},
    )
    preset_2 = EnergyBaselinePreset(
        alignment_mode="REF_GROUP",
        elements=["H", "O"],
        group_to_ref={"A/1": [0.5, 0.5]},
        group_patterns=[],
        config_to_group={"A/1": "A/1"},
    )

    apply_energy_baseline([structure], preset_1)
    energy_after_first_shift = float(structure.energy)
    apply_energy_baseline([structure], preset_2)

    assert float(structure.additional_fields["energy_original"]) == 10.0
    assert float(structure.energy) != 10.0
    assert float(structure.energy) != energy_after_first_shift
