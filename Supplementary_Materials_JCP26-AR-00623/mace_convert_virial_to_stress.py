import os
import numpy as np
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator


def mace_convert_virial_to_stress(
    atoms: list[Atoms], ref_virial_name: str, out_file_name: str
) -> None:
    """
    Convert a virial vector into a stress tensor.

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        input structures
    ref_virial_name: str
        virial label
    out_file_name: str
        name of output file
    """
    formatted_atoms = []
    for at in atoms:
        if ref_virial_name in at.info:
            at.info["REF_stress"] = -at.info[ref_virial_name] / at.get_volume()
            del at.info[ref_virial_name]
            formatted_atoms.append(at)

    write(out_file_name, formatted_atoms, format="extxyz")


if __name__ == '__main__':
    print("Start to convert virials to stress.")
    # atoms: Atoms = read('train.extxyz')
    atoms = read('train.extxyz', index=':')
    ref_virial_name = 'REF_virial'
    out_file_name = 'train_mace.extxyz'
    mace_convert_virial_to_stress(atoms, ref_virial_name, out_file_name)

    # atoms: Atoms = read('test.extxyz')
    atoms = read('test.extxyz', index=':')
    ref_virial_name = 'REF_virial'
    out_file_name = 'test_mace.extxyz'
    mace_convert_virial_to_stress(atoms, ref_virial_name, out_file_name)
    print("Convert virials to stress successfully.")
