from typing import TypedDict, TypeAlias, cast

from rdkit import Chem

from .utils import mol_to_SMILES
from .descriptor import get_BCUT_descriptor
from .constants import METALLIC_ATOM_NUMBERS, ORGANIC_ATOM_NUMBERS

__all__ = ["read_organometal", "is_organometal"]


class Organometal(TypedDict):
    central_metal: str
    ligands: list[str]
    molecule_identifier: list[int | float]


MatchedGroupAtomIdxes: TypeAlias = tuple[int, ...]
MatchedGroupResults: TypeAlias = tuple[MatchedGroupAtomIdxes, ...]


def is_organometal(mol: Chem.Mol):
    num_organic_atoms = 0
    num_metallic_atoms = 0
    num_other_atoms = 0

    for atom in mol.GetAtoms():
        target_atomic_num = atom.GetAtomicNum()
        if target_atomic_num in METALLIC_ATOM_NUMBERS:
            num_metallic_atoms += 1
        elif target_atomic_num in ORGANIC_ATOM_NUMBERS:
            num_organic_atoms += 1
        else:
            num_other_atoms += 1

    if num_metallic_atoms == 1:
        if num_organic_atoms > 1:
            if num_other_atoms == 0:
                return True

    return False


def fragment_central_metal(mol: Chem.Mol):
    bond_indices: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in METALLIC_ATOM_NUMBERS:
            bonds = cast(tuple[Chem.Bond, ...], atom.GetBonds())
            for bond in bonds:
                bond_indices.append(bond.GetIdx())

    f_mol = Chem.FragmentOnBonds(mol, bond_indices)
    f_mol = Chem.RWMol(f_mol)
    for atom_index in range(f_mol.GetNumAtoms() - 1, mol.GetNumAtoms() - 1, -1):
        atom = f_mol.GetAtomWithIdx(atom_index)
        neighbors = cast(tuple[Chem.Atom, ...], atom.GetNeighbors())
        neighbor = neighbors[0]
        if neighbor.GetAtomicNum() in METALLIC_ATOM_NUMBERS:
            f_mol.RemoveAtom(atom_index)
        else:
            atom.SetAtomicNum(109)
            atom.SetIsotope(0)

    mols = cast(
        tuple[Chem.Mol, ...],
        Chem.GetMolFrags(f_mol, asMols=True, sanitizeFrags=False),
    )

    return mols


def standardize_ligand(mol: Chem.Mol):
    have_Mt = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 109:
            have_Mt = True
            break

    # 금속과의 연결이 없을 경우 전하를 통해서 금속 추정
    if not have_Mt:
        charged_atom_idxes: list[int] = []
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() == -1:
                charged_atom_idxes.append(atom.GetIdx())
        if len(charged_atom_idxes) == 1:
            rw_mol = Chem.RWMol(mol)
            mt_idx = rw_mol.AddAtom(Chem.Atom(109))
            rw_mol.AddBond(charged_atom_idxes[0], mt_idx, Chem.BondType.SINGLE)
            mol = rw_mol.GetMol()
        else:
            raise ValueError(
                "Received a ligand without metal connectivity information: "
                f"{Chem.MolToSmiles(mol)}"
            )

    # Beta ketonate 전처리
    beta_ketonate_matched: MatchedGroupResults = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[O;!$([O]~[Mt])]=CC=CO[Mt]")
    )
    if beta_ketonate_matched:
        rw_mol = Chem.RWMol(mol)
        oxygen_index = beta_ketonate_matched[0][0]
        mt_index = rw_mol.AddAtom(Chem.Atom(109))
        rw_mol.AddBond(oxygen_index, mt_index, Chem.BondType.SINGLE)

        return rw_mol.GetMol()

    # 카르보닐기 전처리
    carbonyl_matched: MatchedGroupResults = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[Mt]=C=[O,O+]")
    )
    if carbonyl_matched:
        return Chem.MolFromSmiles("[Mt]C#[O+]", sanitize=False)

    # Cyclopentadienyl기 전처리
    cp_matched: MatchedGroupResults = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[Mt]c1cccc1")
    )
    if cp_matched:
        rw_mol = Chem.RWMol(mol)
        for matched_atom_idx in cp_matched[0]:
            atom = rw_mol.GetAtomWithIdx(matched_atom_idx)
            if atom.GetAtomicNum() == 109:
                continue

            atom.SetIsAromatic(False)
            bonds = cast(tuple[Chem.Bond, ...], atom.GetBonds())
            for bond in bonds:
                if bond.GetIsAromatic():
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)

            neighbors = cast(tuple[Chem.Atom, ...], atom.GetNeighbors())
            has_mt_neighbor = False
            for neighbor in neighbors:
                if neighbor.GetAtomicNum() == 109:
                    has_mt_neighbor = True
                    break

            if has_mt_neighbor:
                continue

            mt_index = rw_mol.AddAtom(Chem.Atom(109))
            rw_mol.AddBond(matched_atom_idx, mt_index, Chem.BondType.SINGLE)

        return rw_mol.GetMol()

    # Arene기 전처리
    arene_matched: MatchedGroupResults = mol.GetSubstructMatches(
        Chem.MolFromSmarts("[Mt]c1ccccc1")
    )
    if arene_matched:
        rw_mol = Chem.RWMol(mol)
        for matched_atom_idx in arene_matched[0]:
            atom = rw_mol.GetAtomWithIdx(matched_atom_idx)
            if atom.GetAtomicNum() == 109:
                continue

            atom.SetIsAromatic(False)
            bonds = cast(tuple[Chem.Bond, ...], atom.GetBonds())
            for bond in bonds:
                if bond.GetIsAromatic():
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)

            neighbors = cast(tuple[Chem.Atom, ...], atom.GetNeighbors())
            has_mt_neighbor = False
            for neighbor in neighbors:
                if neighbor.GetAtomicNum() == 109:
                    has_mt_neighbor = True
                    break

            if has_mt_neighbor:
                continue

            mt_index = rw_mol.AddAtom(Chem.Atom(109))
            rw_mol.AddBond(matched_atom_idx, mt_index, Chem.BondType.SINGLE)

        return rw_mol.GetMol()

    return mol


def read_organometal(SMILES: str) -> Organometal:
    central_metal = ""
    central_metal_atomic_number = 0
    combined_ligand = Chem.Mol()
    ligands: list[str] = []

    mol = Chem.MolFromSmiles(SMILES, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    if not is_organometal(mol):
        raise ValueError(f"Invalid organometal SMILES {SMILES!r}")

    fragments = fragment_central_metal(mol)
    for fragment in fragments:
        if fragment.GetNumAtoms() == 1:
            atomic_number = fragment.GetAtomWithIdx(0).GetAtomicNum()
            if atomic_number in METALLIC_ATOM_NUMBERS:
                central_metal = f"[{fragment.GetAtomWithIdx(0).GetSymbol()}]"
                central_metal_atomic_number = atomic_number
                continue

        standardized_ligand = standardize_ligand(fragment)
        combined_ligand = Chem.CombineMols(combined_ligand, standardized_ligand)
        ligand_SMILES = mol_to_SMILES(standardized_ligand)
        ligands.append(ligand_SMILES)

    bcut_min, bcut_max = get_BCUT_descriptor(combined_ligand)
    molecule_identifier: list[int | float] = [
        central_metal_atomic_number,
        len(ligands),
        round(bcut_min, 8),
        round(bcut_max, 8),
    ]

    return {
        "central_metal": central_metal,
        "ligands": ligands,
        "molecule_identifier": molecule_identifier,
    }
