import re

from rdkit import Chem

__all__ = ["mol_to_SMILES"]


def mol_to_SMILES(mol: Chem.Mol) -> str:
    """Convert a molecule to a simplified, non-isomeric ligand string.

    Removes formal-charge text, atom-map labels, and selected bracketed
    hydrogen annotations (H, H2, and H3). Bare C, N, O, P, S, c, n, and o
    atom brackets are also removed. Mt attachment markers are retained.

    Args:
        mol: RDKit molecule to serialize without modifying it.

    Returns:
        A SMILES-derived string without stereochemical or isotope information.
    """

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
