"""
Defines the ligand PyTorch Dataset class to be used to load training and
validation examples when running the network. No processing or filtering is
done through this class, since it is done in the 00_* scripts.
"""


import torch
import numpy as np
import pandas as pd
from progressbar import progressbar
from torch_geometric.data import Data, Dataset


all_targets = pd.read_csv("../data/dud-e_targets.csv").target_name.tolist()
all_targets = [target.lower() for target in all_targets]

ligand_responses = pd.read_pickle("../data/raw/ligand_responses.pkl")

class LigandDataset(Dataset):
    def __init__(
        self,
        neighborhood_radius,
        negative_ratio,
        targets,
        force_reprocess,
        root="../data/",
        transform=None,
        pre_transform=None,
    ):
        self.force_reprocess = force_reprocess
        self.undersample_to = negative_ratio
        self.neighborhood_radius = neighborhood_radius
        self.targets = targets
        self.set_ligand_dict()
        self.set_ligand_names()
        super(LigandDataset, self).__init__(root, transform, pre_transform)

    def set_ligand_dict(self):
        """Set the ligand codes associated with each target, after resampling."""
        np.random.seed(1)
        self.ligand_dict = dict()
        # For each target, include all of the actives, and resample the decoys
        # if required to.
        for target in self.targets:
            ligand_names = [
                ligand
                for ligand, response in ligand_responses[target].items()
                if response
            ]
            neg_names = [
                ligand
                for ligand, response in ligand_responses[target].items()
                if not response
            ]
            if self.undersample_to is not None:
                ligand_names += np.random.choice(
                    neg_names,
                    size=int(self.undersample_to * len(ligand_names)),
                    replace=False,
                ).tolist()
            else:
                ligand_names += neg_names
            # Set the list of ligand names to the target key in the ligand
            # dictionary.
            self.ligand_dict[target] = ligand_names

    @property
    def raw_file_names(self):
        """Return the raw file names."""
        # The raw files required are the dictionaries containing the ligand
        # Data objects for each target.
        raw_files = [f"{target}_ligand_dict.pkl" for target in all_targets]
        return raw_files

    def set_ligand_names(self):
        """Set the ligand file names to be written to disk."""
        np.random.seed(1)
        self.ligand_names = []
        for target, names in self.ligand_dict.items():
            # The names are in the format target_name.
            self.ligand_names += [f"{target}_{name}.pt" for name in names]
        # Shuffle the names, so ligand Data files will be called in a random
        # order when the network is being trained.
        np.random.shuffle(self.ligand_names)

    def len(self):
        """Return the length of the Dataset's ligand list."""
        return len(self.ligand_names)

    @property
    def processed_file_names(self):
        """Return the name of the processed ligand files."""
        if self.force_reprocess:
            return ["token"]
        return self.ligand_names

    def download(self):
        """All raw files should be generated using 00_* before this script is run."""
        raise Exception(
            "Re-run protein processing script to create {target}_ligand_dict.pkl"
        )

    def process(self):
        """Save each selected ligand as a .pt object."""
        for target in progressbar(self.targets):
            # Read in the list of all ligands for the target.
            curr_ligand_dict = pd.read_pickle(f"../data/raw/{target}_ligand_dict.pkl")
            # Save the selected ones as a .pt object.
            for ligand_name in self.ligand_dict[target]:
                torch.save(
                    curr_ligand_dict[ligand_name],
                    f"../data/processed/{target}_{ligand_name}.pt",
                )

    def get(self, i):
        """Return the ith ligand Data object in the ligand name list."""
        return torch.load(f"../data/processed/{self.ligand_names[i]}")