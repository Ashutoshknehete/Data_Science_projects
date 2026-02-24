import chgnet
from chgnet.model import CHGNet
from pymatgen.core.structure import Structure
import numpy as np
from pymatgen.core import Structure
import sys
import json
import pickle
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.trainer import Trainer
from sklearn.model_selection import train_test_split

JSON_PATH = "/home/cbartel/noord014/projects/ocp/data/oc22/oc22_all.json"
PICKLE_PATH = "/home/cbartel/noord014/projects/deep-learning/scripts/clustering/dataloaders/"

def main():

    chgnet = CHGNet.load()

    with open(JSON_PATH, "r") as file:
        print("Loading data...")
        sys.stdout.flush()
        data = json.load(file)
        print("Data loaded...")
        sys.stdout.flush()

    structure_array = []
    e_per_atom_array = []
    force_array = []
    a_array = []
    b_array = []
    c_array = []
    alpha_array = []
    beta_array = []
    gamma_array = []
    volume_array = []
    formula_array = []

    print("Building featurized vectors...")
    sys.stdout.flush()
    for key, value in data.items():
        if "step_0" in value:
            if value["step_0"]["e_per_at"] < 5 and value["step_0"]["e_per_at"] > -5:
                struc_dict = value["step_0"]["structure"]
                struc_object = Structure.from_dict(struc_dict)
                structure_array.append(struc_object)
                e_dict = value["step_0"]["e_per_at"]
                e_per_atom_array.append(e_dict)
                force_dict = value["step_0"]["forces"]
                force_array.append(force_dict)

                lattice_dict = struc_dict["lattice"]
                a_array.append(lattice_dict["a"])
                b_array.append(lattice_dict["b"])
                c_array.append(lattice_dict["c"])
                alpha_array.append(lattice_dict["alpha"])
                beta_array.append(lattice_dict["beta"])
                gamma_array.append(lattice_dict["gamma"])
                volume_array.append(lattice_dict["volume"])
                formula_array.append(struc_object.formula)
    
    print("Vectors built.")
    sys.stdout.flush()
                
    dataset = pd.DataFrame(
        {
            "a": a_array,
            "b": b_array,
            "c": c_array,
            "alpha": alpha_array,
            "beta": beta_array,
            "gamma": gamma_array,
            "volume": volume_array,
        }
    )

    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(dataset)
    num_clusters = 6

    print("Clustering...")
    sys.stdout.flush()
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=17)
    dataset["cluster"] = kmeans.fit_predict(scaled_dataset)

    # Calculate silhouette score for each data point
    silhouette_avg = silhouette_score(scaled_dataset, dataset["cluster"])
    silhouette_values = silhouette_samples(scaled_dataset, dataset["cluster"])

    unique_clusters = dataset["cluster"].unique()
    # Initialize separate arrays to store row indices for each cluster
    cluster_indices = {cluster: [] for cluster in unique_clusters}
    # Populate the arrays with row indices
    for index, row in dataset.iterrows():
        cluster_indices[row["cluster"]].append(index)

    main_structure_array = structure_array
    all_e_per_atom = e_per_atom_array
    all_forces = force_array

    print("Stratified sampling...")
    sys.stdout.flush()

    # Perform stratified sampling
    train_indices, test_indices = train_test_split(
        dataset.index, test_size=0.2, stratify=dataset["cluster"]
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, stratify=dataset.loc[train_indices]["cluster"]
    )
    
    print("Creating datasets and loaders...")
    sys.stdout.flush()

    # Create train, validation, and test datasets
    train_dataset = StructureData(
        structures=[main_structure_array[index] for index in train_indices],
        energies=[all_e_per_atom[index] for index in train_indices],
        forces=[all_forces[index] for index in train_indices],
    )
    val_dataset = StructureData(
        structures=[main_structure_array[index] for index in val_indices],
        energies=[all_e_per_atom[index] for index in val_indices],
        forces=[all_forces[index] for index in val_indices],
    )
    test_dataset = StructureData(
        structures=[main_structure_array[index] for index in test_indices],
        energies=[all_e_per_atom[index] for index in test_indices],
        forces=[all_forces[index] for index in test_indices],
    )

    batch_sizes = [8,16,32,64]

    for batch_size in batch_sizes:
        train_loader, _, _ = get_train_val_test_loader(
        train_dataset, batch_size=batch_size, train_ratio=1.0, val_ratio=0
        )
    
        val_loader, _, _ = get_train_val_test_loader(
        val_dataset, batch_size=batch_size, train_ratio=1.0, val_ratio=0
        )

        test_loader, _, _ = get_train_val_test_loader(
        test_dataset, batch_size=batch_size, train_ratio=1.0, val_ratio=0
        )

        print(f"Saving dataloaders of batch size: {batch_size}...")
        sys.stdout.flush()

        with open(os.path.join(PICKLE_PATH, str(batch_size), 'train_loader.pkl'), 'wb') as f:
            pickle.dump(train_loader, f)

        with open(os.path.join(PICKLE_PATH, str(batch_size), 'val_loader.pkl'), 'wb') as f:
            pickle.dump(val_loader, f)

        with open(os.path.join(PICKLE_PATH, str(batch_size), 'test_loader.pkl'), 'wb') as f:
            pickle.dump(test_loader, f)
    return None


if __name__ == "__main__":
    main()
