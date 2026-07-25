import numpy as np
from types import SimpleNamespace

from NepTrainKit.core.audit.extract import records_from_result_data, records_from_structures
from NepTrainKit.core.structure import Structure


def _structure(species, *, energy=None, config_type="", forces=None):
    positions = np.arange(len(species) * 3, dtype=np.float64).reshape(-1, 3) * 0.1
    atomic_properties = {"pos": positions, "species": np.asarray(species)}
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    if forces is not None:
        atomic_properties["force"] = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        properties.append({"name": "force", "type": "R", "count": 3})
    additional_fields = {"Config_type": config_type}
    if energy is not None:
        additional_fields["energy"] = energy
    return Structure(np.eye(3) * 5.0, atomic_properties, properties, additional_fields)


def test_records_from_structures_extracts_composition_and_labels():
    structures = [
        _structure(["Fe", "Ni"], energy=-2.0, config_type="bulk", forces=[[1, 0, 0], [0, 2, 0]]),
        _structure(["Fe", "Fe"], energy=-4.0, config_type="defect", forces=[[0, 0, 0], [0, 0, 3]]),
    ]

    records = records_from_structures(structures)

    assert [record.index for record in records] == [0, 1]
    assert records[0].formula == "FeNi"
    assert records[0].composition == {"Fe": 0.5, "Ni": 0.5}
    assert records[0].config_type == "bulk"
    assert records[0].energy_per_atom == -1.0
    assert records[0].max_force == 2.0
    assert records[1].composition == {"Fe": 1.0}
    assert records[1].energy_per_atom == -2.0
    assert records[1].max_force == 3.0


def test_records_from_result_data_preserves_original_indices():
    structures = [
        _structure(["Fe"], config_type="removed"),
        _structure(["Ni"], config_type="active"),
    ]
    result_data = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=np.asarray(structures, dtype=object),
            now_indices=np.asarray([1], dtype=np.int32),
        )
    )

    records = records_from_result_data(result_data)

    assert len(records) == 1
    assert records[0].index == 1
    assert records[0].config_type == "active"
