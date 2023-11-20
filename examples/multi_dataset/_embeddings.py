from typing import List, Optional
import numpy as np
from mdtraj.core.topology import Residue, Chain

opep_embedding_map = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}

opep_termini_embedding_map = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
    "N_N": 25,
    "C_O": 26,
}

embed2res = {value: key for key, value in opep_embedding_map.items()}
all_res = sorted(
    [
        res
        for res in opep_embedding_map.keys()
        if res
        not in [
            "NLE",
            "N",
            "CA",
            "C",
            "O",
            "N_N",
            "C_O",
        ]
    ]
)
cb_types = np.arange(1, 21)


def generate_embeddings(
    embedding_strategy: str,
    residues: List[Residue],
    chains: Optional[List[Chain]] = None,
) -> List[int]:
    """
    Parameters
    ----------
    embedding_strategy:
        "opep" or "opep_termini"
    residues:
        List of mdtraj residues
    chains:
        Optional list of mdtraj chains for systems with multiple molecules

    Returns
    -------
    embeddings:
        list of CG types according to the OPEP CG map
    """
    embeddings = []

    if embedding_strategy not in ["opep", "opep_termini"]:
        raise ValueError(
            "Embedding strategy {} not recognized.".format(embedding_strategy)
        )

    if embedding_strategy == "opep":
        embedding_map = opep_embedding_map
        for res in residues:
            identity = embedding_map[res.name]
            if identity == 6:
                residue_embeddings = [21, 6, 23, 24]
            else:
                residue_embeddings = [21, 22, identity, 23, 24]

            embeddings += residue_embeddings

    if embedding_strategy == "opep_termini":
        embedding_map = opep_termini_embedding_map
        if chains == None:
            for idx, res in enumerate(residues):
                identity = embedding_map[res.name]
                if idx == 0:  # N terminus
                    if identity == 6:
                        residue_embeddings = [25, 6, 23, 24]
                    else:
                        residue_embeddings = [25, 22, identity, 23, 24]

                elif idx == len(residues) - 1:  # C terminus
                    if identity == 6:
                        residue_embeddings = [21, 6, 23, 26]
                    else:
                        residue_embeddings = [21, 22, identity, 23, 26]
                else:
                    if identity == 6:
                        residue_embeddings = [21, 6, 23, 24]
                    else:
                        residue_embeddings = [21, 22, identity, 23, 24]

                embeddings += residue_embeddings
        if chains != None:
            for chain in chains:
                # for now, overwrite input residue list so that each molecule is handled independently
                residues = list(chain.residues)
                # clip caps
                residues = [
                    res
                    for res in residues
                    if res.name not in ["ACE", "NME", "NHE"]
                ]
                for idx, res in enumerate(residues):
                    identity = embedding_map[res.name]
                    if idx == 0:  # N terminus
                        if identity == 6:
                            residue_embeddings = [25, 6, 23, 24]
                        else:
                            residue_embeddings = [25, 22, identity, 23, 24]

                    elif idx == len(residues) - 1:  # C terminus
                        if identity == 6:
                            residue_embeddings = [21, 6, 23, 26]
                        else:
                            residue_embeddings = [21, 22, identity, 23, 26]
                    else:
                        if identity == 6:
                            residue_embeddings = [21, 6, 23, 24]
                        else:
                            residue_embeddings = [21, 22, identity, 23, 24]

                    embeddings += residue_embeddings

    return embeddings
