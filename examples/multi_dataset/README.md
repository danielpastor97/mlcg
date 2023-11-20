# Generating Transferable Force Fields From Diverse Molecular Datasets
----------------------------------------------------------------------

### How To

Generating a diverse CG dataset involves several distinct steps. The script
`generator_routines.py` handles these steps. All configurable options for data
generation are specified in a YAML config file. Please use:

`python prior_generator.py --help`

For documentation

#### 1) Loading and projecting all-atom coordinates and forces

Command:

`python prior_generator.py --save --config PATH_TO_CONFIG`

This procedure loops over the sub datasets specified in the config file, and for
each molecule saves the cg forces, coordinates, embedding inputs (eg, CG
atom types), and  according to the specified
force aggregation strategy. Use of simple force aggregation is recommended.

#### 2) Fitting of CG Prior Terms

Command:

`python prior_generator.py --fit --config PATH_TO_CONFIG`

This procedure takes the projected data and fits to the prior options specified
in the config file. In the end, a prior model (saved as a torch pickle) is saved
with the fitted terms so that it can be used to create delta forces, simulate a
prior-only model, or be combined with a network model
This procedure also generates prior neighborlists for each
molecule in the dataset, which are saved as pickles of `Dict[str,
torch.tensors]`.

#### 3) Subtraction of the Prior Forces and Creation of the Delta Dataset

Command:

`python prior_generator.py --produce --config PATH_TO_CONFIG`

This process uses the fitted prior to subtract prior forces from the full
projected CG forces in order to create a delta dataset for training a neural
network model. From here, everything should be ready for training (eg, input CG
coordinates & atom types, cg delta forces. These can be turned into an h5
dataset using MLCG dataset tools for simple training with YAML configuration
files.

#### 4) Generation of configurations for new molecules.

After training a transferable force field, you can assess its simulation
perfromance on an unseen molecule (eg, one not in the model training/validation
set). First, you must take a CG PDB (one that contains only `N, CA, CB, C,` and
`O` atoms) and ensure that the atoms are in the correct order. To do this, you
may use `reorder_cg_pdb.py` (run `--help` for more information). Once you have
a CG PDB in the correct atom order, you can generate starting configurations for
simulation. It is CRUCIAL that the method used to generate neighborlists for the
training/validation molecules is used in exactly the same way for the
extrapolation molecule. This can be done by using the
`gen_mol_configurations.py` script (again, see `--help` for more information)
and using the same dataset config file that was used to generate the
transferable dataset and priors. Once the configurations have been generated,
they can be simulated using the normal MLCG simulation tools/pipeline.

IMPORTANT: If you are extrapolating to larger molecules, please set the
`max_num_neighbor` SchNet attribute in your model accordingly.

### CAVEATS

All cg topologies are STRICTLY generated to be in the following atom order for each
residue: `N,CA,CB,C,O`. The current tools correct for ordering discrepencies
in AMBER PRO entries in PDBs. Please check the save CG topologies and
coordinates to make sure the correct structure is preserved.

Prior fitting can be sensitive and important. Great care must be taken to ensure
the quality of the prior is good, as neural networks can extrapolate wildly in
data-poor regions (leading to simulation instabilities or sink states). We recommend that users visually
inspect prior energy profiles/parameters and run prior-only simulations of
target molecules to catch any fitting instabilities to data gaps before training
any network models. If the prior
mismatches from the CG projected forces by too much, then the learning can be
disturbed and the model may even experience poor extrapolation or instabilities
during CG simulation. We recommend trying simple prior terms first, and then
systematically adding new priors one by one. It may also be necessary to adjust
a few specified prior constants (eg, certain repulsion excluded volumes) after
fitting if they produce unstable CG simulations. We have only found one instance
where this post-hoc correction is necessary (CA-O repulsion parameters) in our current models.

The final prior model that is used to produce delta forces for training a network
model is the ONLY prior model that can later be combined with said network model
for CG simulations; NO modifcations can be made to the prior after delta forces
have been produced. If a change to the prior is desired, a new delta dataset and
a new network model must be created and trained from scratch respectively. If
your combined network + prior model displays
instabilities during simulation, we recommend to double check that the prior
combined with the network exactly matches the one used to create the delta
dataset.
