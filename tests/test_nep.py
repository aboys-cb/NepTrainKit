#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
import os
from NepTrainKit.core.io.nep import NepTrainResultData,NepPolarizabilityResultData,NepDipoleResultData
from NepTrainKit.core import Structure,Config
from PySide6.QtWidgets import QApplication

app = QApplication()
Config()


class TestNepTrainResultData( unittest.TestCase):
    def setUp(self):


        self.test_dir = Path(__file__).parent
        self.data_dir=os.path.join(self.test_dir,"data/nep")

        self.train_path=os.path.join(self.data_dir,"train.xyz")
    def tearDown(self):
        pass
    def test_load_train(self):
        """测试结构加载功能"""
        result = NepTrainResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.energy.num, 25)
        self.assertEqual(result.force.num, 6250)
        self.assertEqual(result.stress.num, 25)
        self.assertEqual(result.virial.num, 25)
        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.energy.num, 23)
        self.assertEqual(result.force.num, 5750)
        self.assertEqual(result.stress.num, 23)
        self.assertEqual(result.virial.num, 23)
        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(
            os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(
            os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 23)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):
        result = NepTrainResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.energy.num, 25)
        self.assertEqual(result.force.num, 6250)
        self.assertEqual(result.stress.num, 25)
        self.assertEqual(result.virial.num, 25)
        os.remove(os.path.join(self.data_dir,"energy_train.out"))
        os.remove(os.path.join(self.data_dir,"force_train.out"))
        os.remove(os.path.join(self.data_dir,"stress_train.out"))
        os.remove(os.path.join(self.data_dir,"virial_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))


class TestNepPolarizabilityResultData( unittest.TestCase):
    def setUp(self):


        self.test_dir = Path(__file__).parent
        self.data_dir=os.path.join(self.test_dir,"data/polarizability")
        self.train_path=os.path.join(self.data_dir,"train.xyz")

    def tearDown(self):
        pass
    def test_load_train(self):
        """测试结构加载功能"""
        result = NepPolarizabilityResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.polarizability_diagonal.num, 5768)
        self.assertEqual(result.polarizability_no_diagonal.num, 5768)

        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.polarizability_diagonal.num, 5766)
        self.assertEqual(result.polarizability_no_diagonal.num, 5766)

        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 5766)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):
        result = NepPolarizabilityResultData.from_path(self.train_path)
        result.load()
        os.remove(os.path.join(self.data_dir,"polarizability_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))

class TestNepDipoleResultData(unittest.TestCase):
    def setUp(self):


        self.test_dir = Path(__file__).parent
        self.data_dir=os.path.join(self.test_dir,"data/dipole")
        self.train_path=os.path.join(self.data_dir,"train.xyz")

    def tearDown(self):
        pass
    def test_load_train(self):
        """测试结构加载功能"""
        result = NepDipoleResultData.from_path(self.train_path)
        result.load()
        self.assertEqual(result.dipole.num, 5768)

        result.select([0,1,3])
        self.assertEqual(len(result.select_index),3)
        result.uncheck(0)
        self.assertEqual(len(result.select_index),2)

        self.assertEqual(result.select_index , {1,3})
        result.delete_selected()
        self.assertEqual(len(result.select_index) , 0)
        self.assertEqual(result.dipole.num, 5766)


        result.export_model_xyz(self.data_dir)
        export_good_model = Structure.read_multiple(os.path.join(self.data_dir,"export_good_model.xyz"))
        export_remove_model = Structure.read_multiple(os.path.join(self.data_dir,"export_remove_model.xyz"))

        self.assertEqual(len(export_good_model), 5766)
        self.assertEqual(len(export_remove_model), 2)
        os.remove(os.path.join(self.data_dir,"export_good_model.xyz"))
        os.remove(os.path.join(self.data_dir,"export_remove_model.xyz"))

    def test_load_train2(self):
        result = NepDipoleResultData.from_path(self.train_path)
        result.load()
        os.remove(os.path.join(self.data_dir,"dipole_train.out"))
        os.remove(os.path.join(self.data_dir,"descriptor.out"))


if __name__ == "__main__":
    unittest.main()
    app.exit()