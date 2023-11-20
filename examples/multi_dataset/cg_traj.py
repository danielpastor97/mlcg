import numpy as np
import mdtraj as md
import networkx as nx

from mlcg.cg.projection import build_cg_topology
from mlcg.geometry.topology import get_n_paths, Topology


def build_cg_topology(topology, cg_mapping):
    sel_str = " or ".join(
        [f"resname == {k} and name == {v}" for k, v in cg_mapping.keys()]
    )
    sel_ids = topology.select(sel_str)
    cg_topo = topology.subset(sel_ids)
    if cg_topo.n_bonds == 0:
        cg_topo.create_standard_bonds()
    return cg_topo, sel_ids


def build_cg_traj(traj, cg_mapping):
    topo = traj.topology
    cg_topo, sel_ids = build_cg_topology(topo, cg_mapping)
    cg_traj = md.Trajectory(xyz=traj.xyz[:, sel_ids], topology=cg_topo)
    return cg_traj


def get_cg_embedings(md_topo, cg_mapping):
    types = []
    for at in md_topo.atoms:
        (cg_name, cg_type, _) = cg_mapping.get(
            (at.residue.name, at.name), (None, None, None)
        )
        types.append(cg_type)
    return types


def md_topo2mlcg_topo(
    md_topo: md.Topology,
    types=None,
):
    """Makes MLCG topology with bond and angle edges

    Parameters
    ----------
    md_topo:
        MDtraj topology

    Returns
    -------
    mlcg_topo:
        MLCG CG topology
    """

    if md_topo.n_bonds == 0:
        md_topo.create_standard_bonds()

    conn_mat = nx.adjacency_matrix(md_topo.to_bondgraph())

    mlcg_topo = Topology.from_mdtraj(md_topo)
    if types is not None:
        mlcg_topo.types = types

    # Get full bond/angle sets
    bond_edges = get_n_paths(conn_mat, n=2)
    angle_edges = get_n_paths(conn_mat, n=3)
    # Add bonds/angles to topology
    mlcg_topo.bonds_from_edge_index(bond_edges)
    mlcg_topo.angles_from_edge_index(angle_edges)

    return mlcg_topo


def cg_full_topo(topology, cg_mapping):
    cg_topo, _ = build_cg_topology(topology, cg_mapping)
    types = get_cg_embedings(cg_topo, cg_mapping)
    mlcg_topo = md_topo2mlcg_topo(cg_topo, types)
    return mlcg_topo
