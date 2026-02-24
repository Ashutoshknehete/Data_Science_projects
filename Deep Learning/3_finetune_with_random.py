import chgnet
from chgnet.model import CHGNet
from pymatgen.core.structure import Structure    
import numpy as np
from pymatgen.core import Structure
import sys
import json
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(current_directory, '..', '..', 'data', 'oc22_all.json')
with open(json_path, 'r') as file:
    data = json.load(file)


structure_array = []
e_per_atom_array = []
force_array = []

for key, value in data.items():
    if "step_0" in value:
        if value["step_0"]["e_per_at"] < 3 and value["step_0"]["e_per_at"] > -3:
            struc_dict = value["step_0"]["structure"]
            struc_object = Structure.from_dict(struc_dict)
            structure_array.append(struc_object)
            e_dict = value["step_0"]["e_per_at"]        
            e_per_atom_array.append(e_dict)
            force_dict = value["step_0"]["forces"]      
            force_array.append(force_dict)

from chgnet.data.dataset import StructureData, get_train_val_test_loader

dataset = StructureData(
    structures=structure_array,
    energies=e_per_atom_array,
    forces=force_array,
    stresses=None,  # can be None
    magmoms=None,  # can be None
)

train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, batch_size=2, train_ratio=0.9, val_ratio=0.05)

from chgnet.trainer import Trainer
chgnet = CHGNet.load()


# Optionally fix the weights of some layers
for layer in [
    chgnet.atom_embedding,
    chgnet.bond_embedding,
    chgnet.angle_embedding,
    chgnet.bond_basis_expansion,
    chgnet.angle_basis_expansion,
    chgnet.atom_conv_layers[:-1],
    chgnet.bond_conv_layers,
    chgnet.angle_layers,
]:
    for param in layer.parameters():
        param.requires_grad = False

sys.stdout.flush()

# Define Trainer
trainer = Trainer(
    model=chgnet,
    targets="ef",
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=50,
    learning_rate=1e-3,
    use_device="cpu",
    print_freq=100,
)


trainer.train(train_loader, val_loader, test_loader)






