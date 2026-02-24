import os
import sys
from ase.io import read
from ase.io.jsonio import encode
from pymatgen.io.ase import AseAtomsAdaptor
from pydmclab.utils.handy import read_json, write_json #These imports are from my groups python library, they just read and write a json file

HOME_DIR = "/scratch.global/noord014/dl/data/oc22/"
DATA_DIR = "/scratch.global/noord014/dl/data/oc22/raw_trajs"

def process_traj_folder(folder_path=DATA_DIR, remake=False):
    fjson = os.path.join(HOME_DIR, "oc22_all.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    traj_data = {}
    file_counter = 0

    for filename in os.listdir(folder_path):
        if file_counter%50 == 0:
            print(str(file_counter) + "/56366")
            sys.stdout.flush()
        file_counter += 1

        if filename.endswith('.traj'):
            traj_data[filename] = {}
            file_path = os.path.join(folder_path, filename)
            traj_steps = read(file_path, index=":")

            for i, atoms in zip([0,-1], [traj_steps[0],traj_steps[-1]]):
                failed_to_grab = []
                traj_data[filename][f"step_{i}"] = {}

                atoms_dict = encode(atoms)
                traj_data[filename][f"step_{i}"]["atoms"] = atoms_dict
                traj_data[filename][f"step_{i}"]["num_atoms"] = len(atoms)
                
                try:
                    pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)
                    pymatgen_structure = pymatgen_structure.as_dict()
                except Exception as e:
                    print(f"\n {filename} struct failed")
                    failed_to_grab.append("struc")
                    pymatgen_structure = None

                try:
                    total_energy = atoms.get_total_energy()
                except Exception as e:
                    print(f"\n {filename} e_tot failed")
                    failed_to_grab.append("e_tot")
                    total_energy = 0.0

                try:
                    potential_energy = atoms.get_potential_energy()
                except Exception as e:
                    print(f"\n {filename} e_pot failed")
                    failed_to_grab.append("e_pot")
                    potential_energy = 0.0

                try:
                    kinetic_energy = atoms.get_kinetic_energy()
                except Exception as e:
                    print(f"\n {filename} e_kin failed")
                    failed_to_grab.append("e_kin")
                    kinetic_energy = 0.0

                try:
                    forces_array = atoms.get_forces()
                    forces = forces_array.tolist()
                except Exception as e:
                    print(f"\n {filename} forces failed")
                    failed_to_grab.append("forces")
                    forces = None

                traj_data[filename][f"step_{i}"]["structure"] = pymatgen_structure
                traj_data[filename][f"step_{i}"]["e_tot"] = total_energy
                traj_data[filename][f"step_{i}"]["e_pot"] = potential_energy
                traj_data[filename][f"step_{i}"]["e_kin"] = kinetic_energy
                traj_data[filename][f"step_{i}"]["e_per_at"] = total_energy/len(atoms)
                traj_data[filename][f"step_{i}"]["forces"] = forces
                traj_data[filename][f"step_{i}"]["failed_to_grab"] = failed_to_grab

    print("writing data...")
    sys.stdout.flush()
    write_json(traj_data, fjson)
    return read_json(fjson)

def main():
    res = process_traj_folder(remake=True)
    return None

if __name__ == "__main__":
    main()
