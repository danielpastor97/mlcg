import os
import os.path as osp
from glob import glob
from os.path import isfile, join
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)

from ..utils import tqdm
from ..geometry import Topology
from ..cg import build_cg_matrix, build_cg_topology, CA_MAP
from ..neighbor_list import topology2neighbor_list, atomic_data2neighbor_list
from ..data import AtomicData

class ChignolinDataset(InMemoryDataset):
    r""""""

    def __init__(self, root,  transform=None, pre_transform=None,
                 pre_filter=None):
        self.temperature = 350 # K
        super(ChignolinDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def temperature(self):
        """Temperature used to generate the underlying all-atom data"""
        return 350 # K

    def download(self):
        # Download to `self.raw_dir`.
        url_trajectory = 'http://pub.htmd.org/chignolin_trajectories.tar.gz'
        url_forces = 'http://pub.htmd.org/chignolin_forces_nowater.tar.gz'
        url_coords = 'http://pub.htmd.org/chignolin_coords_nowater.tar.gz'
        url_inputs = 'http://pub.htmd.org/chignolin_generators.tar.gz'
        # path_trajectory = download_url(url_trajectory, self.raw_dir)
        path_inputs = download_url(url_inputs, self.raw_dir)
        path_coord = download_url(url_coords, self.raw_dir)
        path_forces = download_url(url_forces, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['chignolin_generators.tar.gz', 'chignolin_forces_nowater.tar.gz', 'chignolin_coords_nowater.tar.gz']

    @property
    def processed_file_names(self):
        return ['chignolin.pt', 'chignolin.pdb']

    @staticmethod
    def get_data_filenames(coord_dir, force_dir):
        tags = [os.path.basename(fn).replace('chig_coor_','').replace('.npy','') for fn in  os.listdir(coord_dir)]

        coord_fns = {}
        for tag in tags:
            fn = f'chig_coor_{tag}.npy'
            coord_fns[tag] = join(coord_dir, fn)
        forces_fns = {}
        for tag in tags:
            fn = f'chig_force_{tag}.npy'
            forces_fns[tag] = join(force_dir, fn)
        return coord_fns,forces_fns

    def process(self):
        # extract files
        for fn in self.raw_paths:
            extract_tar(fn, self.raw_dir, mode='r:gz')
        coord_dir = join(self.raw_dir, 'coords_nowater')
        force_dir = join(self.raw_dir, 'forces_nowater')

        topology = Topology.from_file('/local_scratch/musil/datasets/chignolin/processed/chignolin.pdb')
        embeddings, cg_matrix, _ = build_cg_matrix(topology, cg_mapping=CA_MAP)

        cg_topo = build_cg_topology(topology, cg_mapping=CA_MAP)
        prior_nls = {
            k: topology2neighbor_list(cg_topo, type=k) for k in ['bonds', 'angles']
        }

        n_beads = cg_matrix.shape[0]
        embeddings = np.array(embeddings, dtype=np.int64)

        coord_fns, forces_fns = self.get_data_filenames(coord_dir, force_dir)

        f_proj = np.dot(np.linalg.inv(np.dot(cg_matrix,cg_matrix.T)), cg_matrix)

        data_list = []
        ii_frame = 0
        for i_traj, tag in enumerate(tqdm(coord_fns, desc='Load Dataset')):
            forces = np.load(forces_fns[tag])
            cg_forces = np.array(np.einsum('mn, ind-> imd', f_proj, forces), dtype=np.float32)

            coords = np.load(coord_fns[tag])
            cg_coords = np.array(np.einsum('mn, ind-> imd',cg_matrix, coords), dtype=np.float32)

            n_frames = cg_coords.shape[0]

            for i_frame in range(n_frames):
                pos = torch.from_numpy(cg_coords[i_frame].reshape(n_beads, 3))
                z = torch.from_numpy(embeddings)
                force = torch.from_numpy(cg_forces[i_frame].reshape(n_beads, 3))

                data = AtomicData.from_points(atomic_types=z, pos=pos, forces=force, neighborlist=prior_nls, traj_id=i_traj, frame_id=i_frame)

                if self.pre_filter != None and not self.pre_filter(data):
                    continue
                if self.pre_transform != None:
                    data = self.pre_transform(data)
                data_list.append(data)
                ii_frame += 1

        datas, slices = self.collate(data_list)

        baseline_model = self.get_baseline_model(datas, n_beads)
        self._remove_baseline_forces(datas, slices, baseline_model)

        torch.save((datas, slices), self.processed_paths[0])