"""UI-independent contracts for selective interface exchange and exact quotas."""

from collections import Counter
from dataclasses import replace
from io import StringIO
import re
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from ase import Atoms
from ase.io import read, write

from NepTrainKit.core.cards.alloy import InterfaceLayerMixOperation, InterfaceLayerMixParams
from NepTrainKit.core.cards.errors import CardOperationError


def _si_gap(atoms_per_layer=192, axis=2):
    symbols, positions, groups = [], [], []
    for layer in range(10):
        for site in range(atoms_per_layer):
            point = [(site % 12) / 12, (site // 12) / max(1, atoms_per_layer // 12), (layer + 0.5) / 10]
            if axis != 2:
                point[axis], point[2] = point[2], point[axis]
            positions.append(point)
            symbols.append("Si" if layer < 5 else ("Ga" if site % 2 == 0 else "P"))
            groups.append(layer)
    atoms = Atoms(symbols, scaled_positions=positions, cell=[[12, 1, 0], [0, 16, 0.5], [0.25, 0, 50]], pbc=[True, False, True])
    atoms.new_array("group", np.asarray(groups, dtype=int))
    return atoms


def _params(**kwargs):
    values = dict(axis="c", auto_position=False, interface_position=0.5,
                  left_layers=5, right_layers=5, concentration=0.2,
                  pair_selection="selected", exchange_pairs=(("Si", "P"),),
                  use_seed=True, seed=0)
    values.update(kwargs)
    return InterfaceLayerMixParams(**values)


def _side_changes(original, output, axis=2):
    """Read changed site IDs directly from the original and resulting structures."""
    changed = original.numbers != output.numbers
    left = original.get_scaled_positions(wrap=True)[:, axis] < 0.5
    return np.flatnonzero(changed & left), np.flatnonzero(changed & ~left)


def _left_transitions(original, output, axis=2):
    left, _right = _side_changes(original, output, axis)
    before = np.asarray(original.get_chemical_symbols())
    after = np.asarray(output.get_chemical_symbols())
    return Counter(zip(before[left], after[left]))


class TestSelectiveInterfaceMix(unittest.TestCase):
    def test_vectorized_interface_search_matches_scalar_contrast_reference(self):
        operation = InterfaceLayerMixOperation()

        def scalar_reference(symbols, species, coords):
            values = np.unique(coords)
            best, position = 0.0, None
            for left, right in zip(values[:-1], values[1:]):
                split = 0.5 * (left + right)
                low, high = coords < split, coords >= split
                hist_l = np.asarray([np.count_nonzero(symbols[low] == element) for element in species], float)
                hist_r = np.asarray([np.count_nonzero(symbols[high] == element) for element in species], float)
                hist_l /= np.count_nonzero(low)
                hist_r /= np.count_nonzero(high)
                contrast = 1.0 - operation._cosine(hist_l, hist_r)
                if contrast > best:
                    best, position = contrast, split
            return best, position

        species = ["Si", "Ga", "P"]
        for seed in range(10):
            rng = np.random.default_rng(seed)
            symbols = np.asarray(rng.choice(species, size=120), dtype=object)
            for coords in (rng.random(120), np.repeat(np.arange(30), 4).astype(float)):
                expected = scalar_reference(symbols, species, coords)
                actual = operation._best_split(symbols, species, coords)
                self.assertAlmostEqual(actual[0], expected[0], places=12)
                self.assertEqual(actual[1], expected[1])

        # A flat Si / GaP boundary and its weakly rumpled counterpart must keep
        # the same compositional criterion, regardless of unique-coordinate count.
        atoms = _si_gap(24)
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        coords = atoms.get_scaled_positions()[:, 2]
        for candidate in (coords, coords + np.random.default_rng(19).normal(0, 1e-5, len(coords))):
            expected = scalar_reference(symbols, species, candidate)
            actual = operation._best_split(symbols, species, candidate)
            self.assertAlmostEqual(actual[0], expected[0], places=12)
            self.assertEqual(actual[1], expected[1])
        self.assertEqual(operation._best_split(symbols, species, np.zeros(len(symbols))), (0.0, None))

    def test_pooled_random_uses_atom_uniform_sampling_not_species_saturation(self):
        operation = InterfaceLayerMixOperation()
        atoms = _si_gap()
        for selection in ("all", "selected"):
            counts = []
            frequencies = np.zeros(960, dtype=int)
            for seed in range(100):
                params = _params(
                    pair_selection=selection,
                    exchange_pairs=(("Si", "Ga"), ("Si", "P")),
                    pair_distribution="random", seed=seed,
                )
                output = operation.run_structure(atoms, params)[0]
                left, right = _side_changes(atoms, output)
                self.assertEqual(len(left), 192)
                self.assertEqual(len(right), 192)
                counts.append(_left_transitions(atoms, output)[("Si", "Ga")])
                right = right - 960
                self.assertEqual(len(np.unique(right)), 192)
                frequencies[right] += 1
            # Hypergeometric(960, 480, 192): mean 96, variance about 38.44.
            # Broad deterministic-seed bounds also reject species-edge saturation
            # (the original bug produced only zero or 192 Ga exchanges).
            self.assertTrue(all(0 < count < 192 for count in counts))
            self.assertTrue(90 < np.mean(counts) < 102, counts)
            self.assertTrue(20 < np.var(counts) < 65, counts)
            self.assertTrue(np.all(frequencies > 0))
            self.assertTrue(np.all(frequencies < 45))

    def test_pooled_random_follows_unequal_species_abundance(self):
        atoms = _si_gap()
        symbols = atoms.get_chemical_symbols()
        for site in range(960, len(atoms)):
            symbols[site] = "P" if site % 4 == 0 else "Ga"
        atoms.set_chemical_symbols(symbols)
        operation = InterfaceLayerMixOperation()
        for selection in ("all", "selected"):
            counts = []
            for seed in range(100):
                output = operation.run_structure(atoms, _params(
                    pair_selection=selection,
                    exchange_pairs=(("Si", "Ga"), ("Si", "P")), seed=seed,
                ))[0]
                counts.append(_left_transitions(atoms, output)[("Si", "Ga")])
                self.assertEqual(Counter(output.get_chemical_symbols()), Counter(symbols))
            # 75% Ga in the right pool should give about 144 Ga, not 96 or 192.
            self.assertTrue(138 < np.mean(counts) < 150, counts)
            self.assertTrue(12 < np.var(counts) < 55, counts)

    def test_md_three_models_have_exact_counts_and_common_left_sites_for_20_seeds(self):
        atoms = _si_gap()
        baseline = np.asarray(atoms.get_chemical_symbols())
        operation = InterfaceLayerMixOperation()
        quotas = (40, 38, 38, 38, 38)
        cases = [((("Si", "P"),), "random", Counter({("Si", "P"): 192})),
                 ((("Si", "Ga"), ("Si", "P")), "equal", Counter({("Si", "Ga"): 96, ("Si", "P"): 96})),
                 ((("Si", "Ga"),), "random", Counter({("Si", "Ga"): 192}))]
        for seed in range(20):
            selected_left = []
            for pairs, distribution, expected in cases:
                params = _params(exchange_pairs=pairs, pair_distribution=distribution, layer_pair_counts=quotas, seed=seed)
                out = operation.run_structure(atoms, params)[0]
                new = np.asarray(out.get_chemical_symbols())
                left, right = _side_changes(atoms, out)
                self.assertEqual(_left_transitions(atoms, out), expected)
                self.assertEqual(Counter(atoms.arrays["group"][left]), Counter(dict(zip(range(4, -1, -1), quotas))))
                self.assertEqual(Counter(atoms.arrays["group"][right]), Counter(dict(zip(range(5, 10), quotas))))
                self.assertEqual(int(np.sum(new != baseline)), 384)
                self.assertEqual(Counter(new), Counter(baseline))
                self.assertEqual(len(left), 192)
                self.assertEqual(len(right), 192)
                selected_left.append(set(left))
                np.testing.assert_array_equal(out.positions, atoms.positions)
                np.testing.assert_array_equal(out.cell.array, atoms.cell.array)
                np.testing.assert_array_equal(out.pbc, atoms.pbc)
                np.testing.assert_array_equal(out.arrays["group"], atoms.arrays["group"])
                if distribution == "equal":
                    for layer, quota in enumerate(quotas, 1):
                        layer_right = right[atoms.arrays["group"][right] == layer + 4]
                        self.assertEqual(Counter(baseline[layer_right]), Counter(Ga=quota // 2, P=quota // 2))
            self.assertEqual(selected_left[0], selected_left[1])
            self.assertEqual(selected_left[0], selected_left[2])
        self.assertEqual(atoms.get_chemical_symbols(), list(baseline))
        self.assertNotIn("interface_mix_audit", atoms.info)

    def test_selected_concentration_denominator_is_full_window_and_seed_repeats(self):
        atoms = _si_gap(24)
        operation = InterfaceLayerMixOperation()
        params = _params(concentration=0.2, num_structures=3, seed=123)
        summary = operation.interface_summary(atoms, params)
        self.assertEqual(summary["n_total"], 240)
        self.assertEqual(summary["pair_capacity"], 60)
        self.assertEqual(summary["pair_counts"], [24] * 3)
        self.assertEqual(summary["effective_concentrations"], [0.2] * 3)
        first = operation.run_structure(atoms, params)
        second = operation.run_structure(atoms, params)
        for left, right in zip(first, second):
            self.assertEqual(left.get_chemical_symbols(), right.get_chemical_symbols())
            self.assertEqual(left.info["Config_type"], right.info["Config_type"])
        baseline = np.asarray(atoms.get_chemical_symbols())
        new = np.asarray(first[0].get_chemical_symbols())
        np.testing.assert_array_equal(new[baseline == "Ga"], baseline[baseline == "Ga"])

    def test_four_outputs_have_independent_reproducible_chemical_sites_and_paired_left_ids(self):
        atoms = _si_gap()
        operation = InterfaceLayerMixOperation()
        shared = _params(num_structures=4, seed=123, layer_pair_counts=(40, 38, 38, 38, 38))
        left_selections = []
        for pairs, distribution in [((("Si", "P"),), "random"),
                                    ((("Si", "Ga"), ("Si", "P")), "equal"),
                                    ((("Si", "Ga"),), "random")]:
            params = replace(shared, exchange_pairs=pairs, pair_distribution=distribution)
            first = operation.run_structure(atoms, params)
            repeated = operation.run_structure(atoms, params)
            self.assertEqual(len(first), 4)
            self.assertEqual([int(re.search(r",s=(\d+)", out.info["Config_type"]).group(1)) for out in first], [123, 124, 125, 126])
            changed_sides = [_side_changes(atoms, out) for out in first]
            selections = [frozenset(left) for left, _right in changed_sides]
            self.assertEqual(len(set(selections)), 4)
            self.assertEqual(len({frozenset(right) for _left, right in changed_sides}), 4)
            left_selections.append(selections)
            for original, rerun in zip(first, repeated):
                self.assertEqual(original.get_chemical_symbols(), rerun.get_chemical_symbols())
                self.assertEqual(original.info["Config_type"], rerun.info["Config_type"])
        self.assertEqual(left_selections[0], left_selections[1])
        self.assertEqual(left_selections[0], left_selections[2])

    def test_random_allowed_pair_graph_handles_shared_resources_at_full_capacity(self):
        atoms = Atoms(["Al", "Al", "Si", "Si", "Ga", "Ga", "P", "P"],
                      scaled_positions=[[i / 4, 0, z] for z in (0.25, 0.75) for i in range(4)], cell=np.eye(3) * 8)
        params = _params(left_layers=1, right_layers=1, concentration=1.0,
                         exchange_pairs=(("Al", "Ga"), ("Al", "P"), ("Si", "Ga")))
        for seed in range(20):
            out = InterfaceLayerMixOperation().run_structure(atoms, replace(params, seed=seed))[0]
            left, right = _side_changes(atoms, out)
            self.assertEqual(len(left), 4)
            self.assertEqual(len(right), 4)
            self.assertTrue(set(_left_transitions(atoms, out)).issubset(set(params.exchange_pairs)))
            before, after = np.asarray(atoms.get_chemical_symbols()), np.asarray(out.get_chemical_symbols())
            self.assertTrue(set(zip(after[right], before[right])).issubset(set(params.exchange_pairs)))
            self.assertEqual(Counter(out.get_chemical_symbols()), Counter(atoms.get_chemical_symbols()))

    def test_all_equal_and_nondefault_axis(self):
        atoms = _si_gap(24, axis=0)
        params = _params(axis="a", pair_selection="all", pair_distribution="equal", concentration=0.2)
        output = InterfaceLayerMixOperation().run_structure(atoms, params)[0]
        self.assertEqual(_left_transitions(atoms, output, axis=0), Counter({("Si", "Ga"): 12, ("Si", "P"): 12}))
        np.testing.assert_array_equal(output.cell.array, atoms.cell.array)
        np.testing.assert_array_equal(output.pbc, atoms.pbc)

    def test_equal_pair_capacity_accounts_for_shared_species(self):
        atoms = Atoms(["Al"] * 4 + ["Si"] * 4 + ["Ga"] * 4 + ["P"] * 4,
                      scaled_positions=[[i / 8, 0, z] for z in (0.25, 0.75) for i in range(8)], cell=np.eye(3) * 8)
        params = _params(left_layers=1, right_layers=1, concentration=0.5,
                         exchange_pairs=(("Al", "Ga"), ("Si", "Ga")), pair_distribution="equal")
        operation = InterfaceLayerMixOperation()
        summary = operation.interface_summary(atoms, params)
        self.assertEqual(summary["pair_capacity"], 4)
        for seed in range(20):
            output = operation.run_structure(atoms, replace(params, seed=seed))[0]
            self.assertEqual(_left_transitions(atoms, output), Counter({("Al", "Ga"): 2, ("Si", "Ga"): 2}))
        with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", side_effect=AssertionError("RNG called")):
            with self.assertRaises(CardOperationError):
                operation.run_structure(atoms, replace(params, concentration=0.75))

    def test_fixed_and_gradient_capacity_truth_table_before_rng(self):
        operation = InterfaceLayerMixOperation()
        atoms = _si_gap(24)
        for mode in ("fixed", "gradient"):
            for feasible in (True, False):
                limit = 0.5 if feasible else 0.6
                params = _params(mode=mode, concentration=limit, gradient_start=0, gradient_end=limit, num_structures=5)
                if feasible:
                    for seed in range(20):
                        outputs = operation.run_structure(atoms, replace(params, seed=seed))
                        self.assertEqual(len(outputs), 5)
                        self.assertTrue(all(np.count_nonzero(out.numbers != atoms.numbers) <= 120 for out in outputs))
                else:
                    for seed in range(20):
                        with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", side_effect=AssertionError("RNG called")):
                            with self.assertRaises(CardOperationError):
                                operation.run_structure(atoms, replace(params, seed=seed))

    def test_invalid_selection_and_quotas_fail_before_rng(self):
        params = _params()
        cases = [replace(params, pair_selection="invalid"), replace(params, pair_distribution="invalid"),
                 replace(params, exchange_pairs=()), replace(params, exchange_pairs=(("Si", "Si"),)),
                 replace(params, exchange_pairs=(("P", "Si"),)), replace(params, exchange_pairs=(("Si", "H"),)),
                 replace(params, exchange_pairs=(("Si", "P"), ("Si", "P"))),
                 replace(params, exchange_pairs="Si-P"),
                 replace(params, layer_pair_counts=(2,)), replace(params, layer_pair_counts=(1, 1, 1, -1, 1)),
                 replace(params, layer_pair_counts=(1, 1, 1, 1, 1.5)),
                 replace(params, layer_pair_counts=(200, 0, 0, 0, 0)),
                 replace(params, mode="gradient", layer_pair_counts=(0, 0, 0, 0, 0)),
                 replace(params, exchange_pairs=(("Si", "Ga"), ("Si", "P")), pair_distribution="equal", layer_pair_counts=(1, 0, 0, 0, 0)),
                 replace(params, exchange_pairs=(("Si", "Ga"), ("Si", "P")), pair_distribution="equal", concentration=0.01)]
        for values in cases:
            with self.subTest(params=values):
                with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", side_effect=AssertionError("RNG called")):
                    with self.assertRaises(CardOperationError):
                        InterfaceLayerMixOperation().run_structure(_si_gap(24), values)

    def test_layer_quotas_override_concentration_and_accept_zero(self):
        atoms = _si_gap(24)
        operation = InterfaceLayerMixOperation()
        params = _params(layer_pair_counts=(4, 2, 2, 2, 2), concentration=1.0)
        summary = operation.interface_summary(atoms, params)
        self.assertEqual(summary["requested_concentrations"], [0.1])
        self.assertEqual(summary["effective_concentrations"], [0.1])
        output = operation.run_structure(atoms, params)[0]
        self.assertEqual(np.count_nonzero(output.numbers != atoms.numbers), 24)
        atoms.info["energy"] = -1.0
        zero = operation.run_structure(atoms, replace(params, layer_pair_counts=(0,) * 5))[0]
        self.assertEqual(zero.get_chemical_symbols(), atoms.get_chemical_symbols())
        self.assertEqual(zero.info["energy"], -1.0)
        self.assertNotIn("interface_mix_audit", zero.info)

    def test_explicit_masses_and_compact_labels_survive_extxyz_export(self):
        atoms = _si_gap(24)
        # Unique mass offsets also check that existing isotope masses move with
        # their chemical element, rather than being reset to tabulated values.
        atoms.set_masses(atoms.get_masses() + np.arange(len(atoms)) / 1000)
        original_masses = atoms.get_masses().copy()
        atoms.new_array("spin", atoms.numbers.astype(float))
        output = InterfaceLayerMixOperation().run_structure(atoms, _params())[0]
        for element in np.unique(atoms.numbers):
            np.testing.assert_array_equal(
                np.sort(output.get_masses()[output.numbers == element]),
                np.sort(original_masses[atoms.numbers == element]),
            )
        unchanged = output.numbers == atoms.numbers
        np.testing.assert_array_equal(output.get_masses()[unchanged], original_masses[unchanged])
        np.testing.assert_array_equal(output.arrays["spin"], output.numbers.astype(float))
        np.testing.assert_array_equal(atoms.get_masses(), original_masses)
        stream = StringIO()
        write(stream, output, format="extxyz")
        header = stream.getvalue().splitlines()[1]
        self.assertNotIn("interface_mix_audit", header)
        self.assertNotIn("\\\"", header)
        self.assertLess(len(header), 500)
        self.assertIn("c=0.2", header)
        self.assertIn("s=0", header)
        self.assertIn("pairs=Si-P", header)
        stream.seek(0)
        restored = read(stream, format="extxyz")
        self.assertEqual(restored.info["Config_type"], output.info["Config_type"])
        np.testing.assert_allclose(restored.get_masses(), output.get_masses())
        np.testing.assert_array_equal(restored.arrays["group"], output.arrays["group"])
        np.testing.assert_array_equal(restored.numbers, output.numbers)

    def test_legacy_audit_is_removed_without_sidecars_or_input_mutation(self):
        atoms = _si_gap(24)
        legacy = '{"pairs":[' + '"old record",' * 5000 + ']}'
        atoms.info["interface_mix_audit"] = legacy
        original = atoms.numbers.copy()
        with tempfile.TemporaryDirectory(prefix="compact-interface-export-") as directory:
            for index, concentration in enumerate((0.0, 0.2)):
                out = InterfaceLayerMixOperation().run_structure(atoms, _params(concentration=concentration))[0]
                self.assertNotIn("interface_mix_audit", out.info)
                self.assertEqual(atoms.info["interface_mix_audit"], legacy)
                target = Path(directory) / f"model-{index}.xyz"
                write(target, out, format="extxyz")
                header = target.read_text(encoding="utf-8").splitlines()[1]
                self.assertNotIn("interface_mix_audit", header)
                self.assertLess(len(header), 500)
                self.assertIn(f"c={concentration:g}", header)
                self.assertIn("s=0", header)
                restored = read(target, format="extxyz")
                np.testing.assert_array_equal(restored.numbers, out.numbers)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["model-0.xyz", "model-1.xyz"])
        np.testing.assert_array_equal(atoms.numbers, original)


class TestCompoundInterfaceMix(unittest.TestCase):
    @staticmethod
    def _gap_bn(*, axis=2, alternating=False, sheared=False):
        symbols, positions, layers = [], [], []
        for layer in range(14):
            elements = ("Ga", "P") if layer < 7 else ("B", "N")
            for site in range(16):
                positions.append([(site % 4) / 4, (site // 4) / 4, (layer + 0.5) / 14])
                symbols.append(elements[(layer if layer < 7 else layer - 7) % 2] if alternating else elements[site % 2])
                layers.append(layer)
        scaled = np.asarray(positions)
        if axis != 2:
            scaled[:, [axis, 2]] = scaled[:, [2, axis]]
        cell = np.array([[12, 1, 0], [2, 16, 2], [4, 3, 50]], float) if sheared else np.diag([40, 40, 50])
        pbc = [True, True, True]
        pbc[axis] = False
        atoms = Atoms(symbols, scaled_positions=scaled, cell=cell, pbc=pbc)
        atoms.new_array("group", np.asarray(layers, dtype=int))
        return atoms

    def test_gap_bn_channels_with_shared_pools_and_nonorthogonal_cells(self):
        all_pairs = (("Ga", "B"), ("Ga", "N"), ("P", "B"), ("P", "N"))
        cases = [("all", distribution, (), 0.2) for distribution in ("random", "equal")]
        cases += [("selected", "random", (pair,), 0.2) for pair in all_pairs]
        for pairs in (
            (("Ga", "B"), ("P", "N")),
            (("Ga", "N"), ("P", "B")),
            (("Ga", "B"), ("Ga", "N")),
            (("Ga", "B"), ("P", "B")),
        ):
            cases += [("selected", distribution, pairs, 0.2) for distribution in ("random", "equal")]
        restricted = (("Ga", "B"), ("Ga", "N"), ("P", "B"))
        cases += [("selected", "random", restricted, 0.2), ("selected", "equal", restricted, 0.3)]
        operation = InterfaceLayerMixOperation()
        for axis, sheared in ((2, False), (2, True), (0, True)):
            atoms = self._gap_bn(axis=axis, sheared=sheared)
            original = np.asarray(atoms.get_chemical_symbols())
            for selection, allocation, pairs, concentration in cases:
                with self.subTest(axis=axis, sheared=sheared, selection=selection, allocation=allocation, pairs=pairs):
                    params = _params(
                        axis=("a", "b", "c")[axis], pair_selection=selection,
                        pair_distribution=allocation, exchange_pairs=pairs, concentration=concentration,
                    )
                    summary = operation.interface_summary(atoms, params)
                    self.assertEqual((summary["n_left"], summary["n_right"]), (80, 80))
                    expected_count = 16 if concentration == 0.2 else 24
                    allowed = set(all_pairs if selection == "all" else pairs)
                    for seed in range(20):
                        output = operation.run_structure(atoms, replace(params, seed=seed))[0]
                        left, right = _side_changes(atoms, output, axis)
                        changed = np.concatenate([left, right])
                        after = np.asarray(output.get_chemical_symbols())
                        self.assertEqual((len(left), len(right)), (expected_count, expected_count))
                        transitions = Counter(zip(original[left], after[left]))
                        self.assertEqual(transitions, Counter(zip(after[right], original[right])))
                        self.assertTrue(set(transitions).issubset(allowed))
                        if allocation == "equal":
                            self.assertEqual(transitions, Counter({pair: expected_count // len(allowed) for pair in allowed}))
                        self.assertTrue(np.all((atoms.arrays["group"][changed] >= 2) & (atoms.arrays["group"][changed] <= 11)))
                        self.assertEqual(Counter(after), Counter(original))
                        np.testing.assert_array_equal(output.positions, atoms.positions)
                        np.testing.assert_array_equal(output.cell.array, atoms.cell.array)
                        np.testing.assert_array_equal(output.pbc, atoms.pbc)
                        np.testing.assert_array_equal(output.arrays["group"], atoms.arrays["group"])
                    repeated = operation.run_structure(atoms, replace(params, seed=19))[0]
                    np.testing.assert_array_equal(repeated.numbers, output.numbers)
                    self.assertEqual(repeated.info["Config_type"], output.info["Config_type"])

    def test_alternating_compound_planes_use_actual_window_compositions(self):
        atoms = self._gap_bn(alternating=True)
        operation = InterfaceLayerMixOperation()
        params = _params(pair_selection="all", exchange_pairs=(), pair_distribution="equal")
        summary = operation.interface_summary(atoms, params)
        # Five pure planes give a 3:2 population in each selected region.
        self.assertEqual(summary["pair_capacity"], 64)
        self.assertAlmostEqual(summary["c_max"], 0.8)
        self.assertEqual(summary["pair_counts"], [16])
        output = operation.run_structure(atoms, params)[0]
        self.assertEqual(_left_transitions(atoms, output), Counter({
            ("Ga", "B"): 4, ("Ga", "N"): 4, ("P", "B"): 4, ("P", "N"): 4,
        }))
        self.assertAlmostEqual(operation.interface_summary(atoms, replace(params, pair_distribution="random"))["c_max"], 1.0)
        quotas = (4, 4, 4, 4, 0)
        # Every individual pure-plane pair lacks three of the four channels.
        with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", side_effect=AssertionError("RNG called")):
            with self.assertRaises(CardOperationError):
                operation.run_structure(atoms, replace(params, layer_pair_counts=quotas))
        random_output = operation.run_structure(atoms, replace(params, pair_distribution="random", layer_pair_counts=quotas))[0]
        left, right = _side_changes(atoms, random_output)
        self.assertEqual(Counter(atoms.arrays["group"][left]), Counter({6: 4, 5: 4, 4: 4, 3: 4}))
        self.assertEqual(Counter(atoms.arrays["group"][right]), Counter({7: 4, 8: 4, 9: 4, 10: 4}))

    def test_four_channel_equal_rejects_indivisible_fixed_and_gradient_counts_before_rng(self):
        atoms = self._gap_bn()
        operation = InterfaceLayerMixOperation()
        params = _params(pair_selection="all", exchange_pairs=(), pair_distribution="equal")
        invalid = (
            replace(params, concentration=0.225),  # 18 pairs cannot be shared by four channels.
            replace(params, mode="gradient", gradient_start=0.2, gradient_end=0.225, num_structures=2),
        )
        for values in invalid:
            with self.subTest(mode=values.mode):
                with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", side_effect=AssertionError("RNG called")):
                    with self.assertRaisesRegex(CardOperationError, "divisible"):
                        operation.run_structure(atoms, values)
        valid = replace(params, mode="gradient", gradient_start=0.2, gradient_end=0.3, num_structures=3)
        self.assertEqual(operation.interface_summary(atoms, valid)["pair_counts"], [16, 20, 24])
        for output, pairs in zip(operation.run_structure(atoms, valid), (16, 20, 24)):
            self.assertEqual(set(_left_transitions(atoms, output).values()), {pairs // 4})


if __name__ == "__main__":
    unittest.main()
