import os
import sys
import torch
import pickle
import chgnet
from chgnet.model import CHGNet
from chgnet.trainer import Trainer

DATA_DIR = "/home/cbartel/noord014/projects/deep-learning/scripts/clustering/dataloaders"
BATCH_SIZE = 32

def load_dataloader_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        dataloader = pickle.load(file)
    return dataloader

def load_loaders():
    print("Loading dataloaders...")
    sys.stdout.flush()

    train_dataloader = load_dataloader_from_pickle(os.path.join(DATA_DIR, str(BATCH_SIZE), "train_loader.pkl"))
    val_dataloader = load_dataloader_from_pickle(os.path.join(DATA_DIR, str(BATCH_SIZE), "val_loader.pkl"))
    test_dataloader = load_dataloader_from_pickle(os.path.join(DATA_DIR, str(BATCH_SIZE), "test_loader.pkl"))

    return train_dataloader, val_dataloader, test_dataloader

def load_chgnet():
    chgnet = CHGNet.load()
    return chgnet

def fix_weights(chgnet):
    print("Fixing weights...")
    sys.stdout.flush()

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
    return chgnet

def load_trainer(chgnet, device="cpu", num_epochs=5, learning_rate=1e-3, print_freq=100):
    print("Loading trainer...")
    sys.stdout.flush()

    trainer = Trainer(
        model=chgnet,
        targets="ef",
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=num_epochs,
        learning_rate=learning_rate,
        use_device=device,
        print_freq=print_freq,
    )
    return trainer

def main():
    train_loader, val_loader, test_loader = load_loaders()
    chgnet = load_chgnet()
    chgnet = fix_weights(chgnet)
    trainer = load_trainer(chgnet, device="cpu", num_epochs=5, learning_rate=1e-3, print_freq=100)
    print("Beginning training...")
    sys.stdout.flush()
    trainer.train(train_loader, val_loader, test_loader)    
    return None

if __name__ == "__main__":
    main()
