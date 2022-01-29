import torch
import numpy as np
import mdtraj
from mdtraj.core.element import carbon
from mlcg.geometry.internal_coordinates import compute_torsions
import copy
from mlcg.geometry.statistics import _get_bin_centers
from mlcg.nn.prior import Dihedrals

# Mock data of ten atoms for 1000 frames
n_frames = 1000
n_atoms = 10
test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))

dihedral_atoms = np.array(
    [[i, i + 1, i + 2, i + 3] for i in range(test_coords.shape[1] - 3)]
)
mapping = torch.tensor(
    np.concatenate(
        [dihedral_atoms.T + (n_atoms * i) for i in range(n_frames)], axis=1
    )
)

topo = mdtraj.core.topology.Topology()
topo.add_chain()
topo.add_residue("ALA1", topo._chains[0])
for _ in range(n_atoms):
    topo.add_atom("CA", carbon, list(topo.residues)[0])
traj = mdtraj.core.trajectory.Trajectory(test_coords, topo)


def test_mdtraj_dihedral_compatibility():
    """Tests to make sure that MLCG torsion calculations
    aggree with the dihedral calculations from mlcg
    """

    mlcg_dihedrals = (
        compute_torsions(torch_coords, mapping)
        .numpy()
        .reshape(n_frames, dihedral_atoms.shape[0])
    )
    mdtraj_dihedrals = mdtraj.compute_dihedrals(traj, dihedral_atoms)
    np.testing.assert_allclose(mlcg_dihedrals, mdtraj_dihedrals, rtol=1e-3)

def test_dihedral_aic_criterion():
    '''
        Test to ensure that proper index of fitting is choosen
        Perturb coordinates of first dihedral with a 3-amplitude sine wave
    '''
    TargetPrior = Dihedrals

    test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
    torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))
    torch_coords[1::n_atoms] = torch.tensor((0,0,0)) 
    torch_coords[2::n_atoms] = torch.tensor((1,0,0)) 
    torch_coords[3::n_atoms] = torch.tensor((2,1,0)) 
    pos = torch_coords[:4]
    ind_mapping = torch.tensor((0,1,2,3))
    ind_mapping = torch.reshape(ind_mapping,[4,1])
    theta_map = theta_mapping(pos,ind_mapping)
    k1s = [1,2,3]
    samples = sample_from_curve(k1s,n_frames)
    for i_s,sample in enumerate(samples):
        for k,v in theta_map.items():
            if np.abs(sample,v) < 1e-2:
                torch_coords[i_s*n_atoms] = torch.stack(k)

    values = TargetPrior.compute_features(torch_coords, mapping)
    b_min = values.min()
    b_max = values.max()
    nbins = 100
    bin_centers = _get_bin_centers(values, nbins, b_min=b_min, b_max=b_max)
    hist = torch.histc(values, bins=nbins, min=b_min, max=b_max)

    mask = hist > 0
    bin_centers_nz = bin_centers[mask]

    ncounts_nz = hist[mask]
    dG_nz = -torch.log(ncounts_nz)
    params = TargetPrior.fit_from_potential_estimates(
        bin_centers_nz, dG_nz
    )
    assert all(params['k1s'][4:] == 0)

def sample_from_curve(k1s,n_frames):
    V = 0
    thetas = torch.from_numpy(np.linspace(-np.pi,np.pi,100))
    for ik,k1 in enumerate(k1s):
        V += k1*torch.sin(ik*thetas)
    pi = torch.exp(-V)
    pi = pi/torch.sum(pi)
    cum_pi = torch.cumsum(pi,dim=0).numpy()
    theta = []
    for _ in range(n_frames):
        sel_ind = np.where(cum_pi>np.random.rand())[0][0]
        theta.append(thetas[sel_ind].numpy())
    return theta

def theta_mapping(pos,mapping):
    theta_map = {}
    r = 1
    thetas = torch.linspace(-np.pi,np.pi,100)
    for thet in thetas:
        x0 = torch.tensor(-1)
        y0 = r*torch.cos(thet)
        z0 = r*torch.sin(thet)
        pos[0] = torch.tensor((x0,y0,z0))
        dr1 = pos[mapping[1]] - pos[mapping[0]]
        dr1 = dr1 / dr1.norm(p=2, dim=1)[:, None]
        dr2 = pos[mapping[2]] - pos[mapping[1]]
        dr2 = dr2 / dr2.norm(p=2, dim=1)[:, None]
        dr3 = pos[mapping[3]] - pos[mapping[2]]
        dr3 = dr3 / dr3.norm(p=2, dim=1)[:, None]
        n1 = torch.cross(dr1, dr2, dim=1)
        n2 = torch.cross(dr2, dr3, dim=1)
        m1 = torch.cross(n1, dr2, dim=1)
        y = torch.sum(m1 * n2, dim=-1)
        x = torch.sum(n1 * n2, dim=-1)
        theta = torch.atan2(-y, x)
        theta_map[(x0,y0,z0)] = theta.numpy()
    return theta_map