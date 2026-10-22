from typing import TypedDict, TypeAlias, cast

import numpy as np
from rdkit import Chem

SMARTS: TypeAlias = str


class GroupMappingResult(TypedDict):
    group_idx: int
    atoms: tuple[int, ...]


def map_groups(
    mol: Chem.Mol,
    groups: list[SMARTS],
    *,
    strict: bool = True,
) -> list[GroupMappingResult]:
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)

    matched_groups: list[GroupMappingResult] = []
    heavy_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    for group_idx, SMARTS in enumerate(groups):
        patt = Chem.MolFromSmarts(SMARTS)
        candidates: tuple[tuple[int, ...], ...] = mol.GetSubstructMatches(patt)
        for candidate in candidates:
            if set(candidate).issubset(heavy_atoms):
                heavy_atoms.difference_update(candidate)
                matched_groups.append({"group_idx": group_idx, "atoms": candidate})

    if strict and len(heavy_atoms) != 0:
        unmatched_SMILES = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=sorted(heavy_atoms),
            canonical=True,
            isomericSmiles=False,
        )
        unmatched_groups = unmatched_SMILES.replace(".", " / ")

        raise ValueError(
            f"Molecule does not consist only of the suggested groups.\nUnmatched atoms: {heavy_atoms}\nUnmatched fragments: {unmatched_groups}\nSMILES: {Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)}"
        )

    return matched_groups


def get_molecular_descriptor(
    mol: Chem.Mol,
    groups: list[SMARTS],
    *,
    strict: bool = True,
):
    """Count greedily mapped occurrences of each molecular group.

    Overlapping matches are not counted independently, and earlier SMARTS
    patterns take precedence.

    Args:
        mol: RDKit molecule to describe.
        groups: SMARTS patterns whose accepted occurrences are counted.
        strict: Whether every heavy atom must be assigned to a group.

    Returns:
        A list of length ``len(groups)``. Element ``i`` is the number of
        accepted matches for ``groups[i]``.

    Raises:
        ValueError: If strict mapping leaves one or more heavy atoms
            unmatched.

    Note:
        Valid SMARTS patterns are required. An invalid pattern can raise an
        underlying RDKit exception.
    """

    group_descriptor = [0] * (len(groups))
    group_mapping_results = map_groups(mol, groups, strict=strict)
    for group_mapping_result in group_mapping_results:
        group_descriptor[group_mapping_result["group_idx"]] += 1

    return np.array(group_descriptor, dtype=np.float64)
