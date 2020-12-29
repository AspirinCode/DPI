# DPI
Drug-protein interaction prediction using graph neural networks.

To use the scripts in `src/` for your own purposes:

1. Download the DUD-E dataset from [here](http://dude.docking.org/db/subsets/all/all.tar.gz).
2. Adjust the paths as required:

   * Run `00_process_ligands.ipynb` to create the PyTorch `Data` objects from the ligand `.mol2` files.
   * Run `00_process_ligands.ipynb` to create the PyTorch `Data` objects from the protein `.pdb` files.
 
3. Use the `LigandDataset.py` module to create two PyTorch `Dataset`s: one named `training_set.pkl`,
   and the other `validation_set.pkl`. The easiest way to do this is to supply lists of DUD-E target
   names corresponding to those used to create training and validation examples respectively.
4. Use `01_network.ipynb` to train and validate a GCN on the training and validation sets. Currently,
   this notebook contains the lists of targets I've used for training and validation for reference.
   
The next major upgrades to be made to this pipeline involve:

1. A framework for hyperparameter tuning.
2. Integration of other data sources into the learning model.

Issues and suggestions can be added to this repo's Issues tab.
