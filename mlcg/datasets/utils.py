import torch
from torch_geometric.loader import DataLoader
from typing import Dict, List, Sequence
from mlcg.data.atomic_data import AtomicData
import mdtraj as md
import numpy as np

from ..data._keys import FORCE_KEY


def chunker(seq: Sequence, size: int):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def remove_baseline_forces(
    data_list: List[AtomicData], models: Dict[str, torch.nn.Module]
):
    """Compute the forces on the input :obj:`data_list` with the :obj:`models`
    and remove them from the reference forces contained in :obj:`data_list`.
    The computation of the forces is done on the whole :obj:`data_list` at once
    so it should not be too large.
    """
    n_frame = len(data_list)
    dataloader = DataLoader(data_list, batch_size=n_frame)
    baseline_forces = []
    for data in dataloader:
        for k in models.keys():
            models[k].eval()

            data = models[k](data)
            baseline_forces.append(data.out[k][FORCE_KEY].flatten())
            # make sure predicted properties don't require gradient anymore
            for key, v in data.out[k].items():
                data.out[k][key] = v.detach()
    baseline_forces = torch.sum(torch.vstack(baseline_forces), dim=0).view(
        -1, 3
    )

    for i_frame in range(n_frame):
        mask = data.batch == i_frame
        data_list[i_frame].forces -= baseline_forces[mask]
        data_list[i_frame].baseline_forces = baseline_forces[mask]

    return data_list


def write_PDB(dataset, frame=0, fout="cg.pdb"):
    """
    Given a mlcg Atomic Data object write out trajectory in PDB for a particular frame
    More or less a copy of mdtraj PDBReporter but explicit writing of bond connection
    """
    topology = dataset.topologies.to_mdtraj()
    n_atoms = topology.n_atoms
    cg_traj = md.Trajectory(
        dataset.data.pos[
            int(frame * n_atoms) : int((frame + 1) * n_atoms)
        ].numpy(),
        topology,
    )
    chains = [chain for chain in topology.chains]
    bfactors = ["{0:5.2f}".format(0.0)] * cg_traj.xyz.shape[1]

    file = open(fout, "w")
    _write_header(fout)
    atomIndex = 1
    posIndex = 0
    modelIndex = 0
    print("MODEL     %4d" % modelIndex, file=file)

    for (chainIndex, chain) in enumerate(topology.chains):
        chainName = chains[chainIndex].index
        # converts int to alphabet (0->a,1->b, etc.) to match convention
        chainName = chr(chainName + 97).upper()
        residues = list(chain.residues)
        for (resIndex, res) in enumerate(residues):
            if len(res.name) > 3:
                resName = res.name[:3]
            else:
                resName = res.name
            for atom in res.atoms:
                if (
                    len(atom.name) < 4
                    and atom.name[:1].isalpha()
                    and (atom.element is None or len(atom.element.symbol) < 2)
                ):
                    atomName = " " + atom.name
                elif len(atom.name) > 4:
                    atomName = atom.name[:4]
                else:
                    atomName = atom.name
                coords = cg_traj.xyz[0, posIndex]
                if atom.element is not None:
                    symbol = atom.element.symbol
                else:
                    symbol = " "
                if atom.serial is not None and len(topology._chains) < 2:
                    # We can't do this for more than 1 chain
                    # to prevent issue 1611
                    atomSerial = atom.serial
                else:
                    atomSerial = atomIndex
                line = (
                    "ATOM  %5d %-4s %3s %1s%4d    %s%s%s  1.00 %5s      %-4s%2s  "
                    % (  # Right-justify atom symbol
                        atomSerial % 100000,
                        atomName,
                        resName,
                        chainName,
                        (res.resSeq) % 10000,
                        _format_83(coords[0]),
                        _format_83(coords[1]),
                        _format_83(coords[2]),
                        bfactors[posIndex],
                        atom.segment_id[:4],
                        symbol[-2:],
                    )
                )
                assert len(line) == 80, "Fixed width overflow detected"
                print(line, file=file)
                posIndex += 1
                atomIndex += 1
            if resIndex == len(residues) - 1:
                print(
                    "TER   %5d      %3s %s%4d"
                    % (atomSerial + 1, resName, chainName, res.resSeq),
                    file=file,
                )
                atomIndex += 1

    _write_footer(file, topology)
    file.close()
    return None


def _write_header(
    file, unitcell_lengths=None, unitcell_angles=None, write_metadata=True
):
    """Write out the header for a PDB file.
    Parameters
    ----------
    unitcell_lengths : {tuple, None}
        The lengths of the three unitcell vectors, ``a``, ``b``, ``c``
    unitcell_angles : {tuple, None}
        The angles between the three unitcell vectors, ``alpha``,
        ``beta``, ``gamma``
    """

    if unitcell_lengths is None and unitcell_angles is None:
        return
    if unitcell_lengths is not None and unitcell_angles is not None:
        if not len(unitcell_lengths) == 3:
            raise ValueError("unitcell_lengths must be length 3")
        if not len(unitcell_angles) == 3:
            raise ValueError("unitcell_angles must be length 3")
    else:
        raise ValueError(
            "either unitcell_lengths and unitcell_angles"
            "should both be spefied, or neither"
        )

    box = list(unitcell_lengths) + list(unitcell_angles)
    assert len(box) == 6

    if write_metadata:
        print("REMARK   ", file=file)
    print(
        "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 " % tuple(box),
        file=file,
    )


def _write_footer(file, topology=None):

    conectBonds = []
    if topology is not None:
        for atom1, atom2 in topology.bonds:
            conectBonds.append((atom1, atom2))
    if len(conectBonds) > 0:

        # Work out the index used in the PDB file for each atom.

        atomIndex = {}
        nextAtomIndex = 0
        prevChain = None
        for chain in topology.chains:
            for atom in chain.atoms:
                if atom.residue.chain != prevChain:
                    nextAtomIndex += 1
                    prevChain = atom.residue.chain
                atomIndex[atom] = nextAtomIndex
                nextAtomIndex += 1

        # Record which other atoms each atom is bonded to.

        atomBonds = {}
        for atom1, atom2 in conectBonds:
            index1 = atomIndex[atom1]
            index2 = atomIndex[atom2]
            if index1 not in atomBonds:
                atomBonds[index1] = []
            if index2 not in atomBonds:
                atomBonds[index2] = []
            atomBonds[index1].append(index2)
            atomBonds[index2].append(index1)

        # Write the CONECT records.

        for index1 in sorted(atomBonds):
            bonded = atomBonds[index1]
            while len(bonded) > 4:
                print(
                    "CONECT%5d%5d%5d%5d"
                    % (index1, bonded[0], bonded[1], bonded[2]),
                    file=file,
                )
                del bonded[:4]
            line = "CONECT%5d" % index1
            for index2 in bonded:
                line = "%s%5d" % (line, index2)
            print(line, file=file)
    print("END", file=file)


def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return "%8.3f" % f
    if -9999999 < f < 99999999:
        return ("%8.3f" % f)[:8]
    raise ValueError(
        'coordinate "%s" could not be represnted ' "in a width-8 field" % f
    )


def write_PSF(dataset, fout="cg.psf", charges=None):
    """
    Write out charmm format psf file from AtomicData object
    """
    file = open(fout, "w")
    print("PSF", file=file)
    cg_topo = dataset.topologies
    topology = cg_topo.to_mdtraj()
    n_atoms = topology.n_atoms

    # Add in dummy charge
    if charges == None:
        charges = ["{0:2.6f}".format(0.0)] * n_atoms
    seg_names = ["{:2s}".format("CG")] * n_atoms

    if hasattr(dataset.data, "masses"):
        masses = dataset.data.masses
    else:
        masses = ["{0:2.4f}".format(0)] * n_atoms

    # Write out atom information
    atom_index = 1
    pos_index = 0
    print("{:>8d} !NATOM".format(n_atoms), file=file)
    for (_, chain) in enumerate(topology.chains):
        # converts int to alphabet (0->a,1->b, etc.) to match convention
        residues = list(chain.residues)
        for (_, res) in enumerate(residues):
            for atom in res.atoms:
                atom_type = atom.name
                atom_name = atom.name
                resid = atom.residue.index
                resname = atom.residue.name
                line = "{:>8d} {:<4s} {:<4d} {:<4} {:<4} {:<4} {} {:>2.4} 0".format(
                    atom_index % 100000,
                    seg_names[pos_index],
                    resid,
                    resname,
                    atom_name,
                    atom_type,
                    charges[pos_index],
                    masses[pos_index],
                )
                print(line, file=file)
                atom_index += 1
                pos_index += 1

    # Write out bonding information
    properties = ["bonds", "angles", "dihedrals", "impropers"]
    titles = [
        "!NBOND: bonds",
        "!NTHETA: angles",
        "!NPHI: dihedrals",
        "!NIMPHI: impropers",
    ]
    items_per_line = [4, 3, 2, 2]

    for i_p, property in enumerate(properties):
        prop = getattr(cg_topo, property)
        prop = np.asarray(prop).T
        n_prop = len(prop)
        title = "{:>8d} ".format(n_prop) + titles[i_p]
        print(title, file=file)
        item_per_line = items_per_line[i_p]
        i_b = 0
        while i_b < n_prop:
            print_list = []
            for _ in range(item_per_line):
                if i_b > n_prop - 1:
                    continue
                for atom in prop[i_b]:
                    print_list.append(atom + 1)
                i_b += 1
            string = ["{:>8d}".format(x) for x in print_list]
            string = "".join(string)
            print(string, file=file)

    file.close()
