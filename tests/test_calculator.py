#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import os
import numpy as np
from pathlib import Path

from NepTrainKit.core import Structure
from NepTrainKit.core.calculator import Nep3Calculator

class TestNep(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent

        self.calculator = Nep3Calculator(os.path.join(self.test_dir,"data/nep/nep.txt"))
        self.structures = Structure.read_multiple(os.path.join(self.test_dir,"data/nep/train.xyz"))[0]
        self.energy = np.load(os.path.join(self.test_dir,"data/nep/energy.npy"))
        self.forces = np.load(os.path.join(self.test_dir,"data/nep/forces.npy"))
        self.virial = np.load(os.path.join(self.test_dir,"data/nep/virial.npy"))

    def tearDown(self):
        pass
    
    def test_initialization(self):
        self.assertTrue(self.calculator.initialized)
        
    def test_calculate(self):
        potentials, forces, virials = self.calculator.calculate(self.structures)
        np.testing.assert_array_equal(self.energy, potentials)
        np.testing.assert_array_equal(self.forces, forces)
        np.testing.assert_array_equal(self.virial, virials)

    def test_get_descriptor(self):
        descriptor = self.calculator.get_descriptor(self.structures)
        local_descriptor = np.load(os.path.join(self.test_dir,"data/nep/descriptor.npy" ))
        np.testing.assert_array_equal(local_descriptor, descriptor)
        
    def test_get_structures_descriptor(self):
        structure_descriptors = self.calculator.get_structures_descriptor(self.structures)

        local_descriptor = np.load(os.path.join(self.test_dir,"data/nep/descriptor.npy" ))
        local_structure_descriptor = np.mean(local_descriptor,axis=0).reshape(-1,structure_descriptors.shape[1])
        np.testing.assert_array_almost_equal(local_structure_descriptor, structure_descriptors,decimal=6)


class TestDipole(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent
        self.calculator = Nep3Calculator(os.path.join(self.test_dir,"data/dipole/nep.txt"))
        self.structures = Structure.read_multiple(os.path.join(self.test_dir,"data/dipole/train.xyz"))[0]

    def tearDown(self):
        pass

    def test_initialization(self):
        self.assertTrue(self.calculator.initialized)

    def test_calculate(self):
        dipole = self.calculator.get_structures_dipole(self.structures)
        local_dipole = np.array([[0.15439024567604065, 0.005705520510673523, 0.0044387467205524445]], dtype=np.float32)
        np.testing.assert_array_equal(local_dipole, dipole)


class TestPolarizability(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent

        self.calculator = Nep3Calculator(os.path.join(self.test_dir,"data/polarizability/nep.txt"))
        self.structures = Structure.read_multiple(os.path.join(self.test_dir,"data/polarizability/train.xyz"))[0]

    def tearDown(self):
        pass

    def test_initialization(self):
        self.assertTrue(self.calculator.initialized)

    def test_calculate(self):
        pol = self.calculator.get_structures_polarizability(self.structures)

        local_pol = np.array([[100.79893493652344, 92.42485046386719, 56.936161041259766, 3.494504451751709, -0.08088953793048859, 0.07827239483594894]], dtype=np.float32)
        np.testing.assert_array_equal(local_pol, pol)


if __name__ == "__main__":
    unittest.main()