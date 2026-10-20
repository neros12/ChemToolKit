import re

from rdkit import Chem

__all__ = ["mol_to_SMILES"]


def mol_to_SMILES(mol: Chem.Mol) -> str:
    def _remove_charge_from_bracket_atom(match: re.Match[str]) -> str:
        atom_text = match.group(1)

        return (
            "[" + re.compile(r"(?:[+-]{1,3}\d*|\d+[+-]{1,3})").sub("", atom_text) + "]"
        )

    SMILES = Chem.MolToSmiles(mol, isomericSmiles=False)
    SMILES = re.compile(r"\[([^\]]*)\]").sub(_remove_charge_from_bracket_atom, SMILES)
    SMILES = re.sub(r":\d+(?=\])", "", SMILES)
    SMILES = SMILES.replace("H]", "]")
    SMILES = SMILES.replace("H2]", "]")
    SMILES = SMILES.replace("H3]", "]")
    SMILES = SMILES.replace("[C]", "C")
    SMILES = SMILES.replace("[N]", "N")
    SMILES = SMILES.replace("[O]", "O")
    SMILES = SMILES.replace("[P]", "P")
    SMILES = SMILES.replace("[S]", "S")
    SMILES = SMILES.replace("[c]", "c")
    SMILES = SMILES.replace("[n]", "n")
    SMILES = SMILES.replace("[o]", "o")
    SMILES = SMILES.replace("[]", "")

    return SMILES
