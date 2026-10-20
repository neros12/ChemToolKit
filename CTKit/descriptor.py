from rdkit import Chem

import numpy as np

__all__ = ["get_BCUT_descriptor", "cosine_similarity"]


def get_BCUT_descriptor(mol: Chem.Mol):
    mol = Chem.RemoveAllHs(mol)
    molecular_matrix = np.full((mol.GetNumAtoms(), mol.GetNumAtoms()), 0.001)

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 109:
            molecular_matrix[atom.GetIdx(), atom.GetIdx()] = 1
        else:
            molecular_matrix[atom.GetIdx(), atom.GetIdx()] = atomic_num

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if bond.GetIsAromatic():
            weight = 0.15
        elif bond.GetBondType() in (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ):
            weight = 0.1 * bond.GetBondTypeAsDouble()
        else:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")

        i = begin.GetIdx()
        j = end.GetIdx()
        molecular_matrix[i, j] = weight
        molecular_matrix[j, i] = weight

    eigenvalues = np.linalg.eigvalsh(molecular_matrix)

    return float(eigenvalues[0]), float(eigenvalues[-1])


def cosine_similarity(
    vector_a: list[float | int] | np.ndarray[tuple[int], np.dtype[np.number]],
    vector_b: list[float | int] | np.ndarray[tuple[int], np.dtype[np.number]],
) -> float:
    array_a = np.asarray(vector_a, dtype=float)
    array_b = np.asarray(vector_b, dtype=float)

    scale_a = np.max(np.abs(array_a))
    scale_b = np.max(np.abs(array_b))
    if scale_a == 0 or scale_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    array_a = array_a / scale_a
    array_b = array_b / scale_b
    similarity = np.dot(
        array_a / np.linalg.norm(array_a),
        array_b / np.linalg.norm(array_b),
    )

    return float(np.clip(similarity, -1.0, 1.0))
