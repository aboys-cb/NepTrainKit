from .card_test_base import *


class MagnetismCardTest(BaseCardTest):
    @staticmethod
    def _spin_chain():
        atoms = Atoms(
            symbols=["Fe", "Fe", "Fe", "Fe"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            cell=np.diag([6.0, 6.0, 8.0]),
            pbc=[False, False, True],
        )
        atoms.info["Config_type"] = "Fe_chain"
        return atoms

    @staticmethod
    def _spin_bilayer_chain():
        atoms = Atoms(
            symbols=["Fe"] * 8,
            positions=[
                [0.0, 0.0, 0.00],
                [1.0, 0.0, 0.01],
                [0.0, 0.0, 1.00],
                [1.0, 0.0, 1.01],
                [0.0, 0.0, 2.00],
                [1.0, 0.0, 2.01],
                [0.0, 0.0, 3.00],
                [1.0, 0.0, 3.01],
            ],
            cell=np.diag([4.0, 4.0, 6.0]),
            pbc=[False, False, True],
        )
        atoms.info["Config_type"] = "Fe_bilayer_chain"
        return atoms
