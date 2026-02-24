import os
import sys
import json
import numpy as np
from ase.io.jsonio import decode
from chgnet.model import CHGNet 
from pymatgen.core.structure import Structure
from pydmclab.utils.handy import read_json, write_json

DATA_DIR = "/scratch.global/noord014/dl/data/oc22"

def get_data():
    fjson = os.path.join(DATA_DIR, "oc22_all.json")
    if os.path.exists(fjson):
        print("loading data...")
        sys.stdout.flush()
        return read_json(fjson)
    else:
        return None

def run_zero_shot(data, remake=False):
    fjson = os.path.join(DATA_DIR, "oc22_zeroshot.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    data_zs = {}
    chgnet = CHGNet.load()
    file_counter = 0

    for filename in data:
        if file_counter%50 == 0:
            print(str(file_counter) + "/56366")
            sys.stdout.flush()
        file_counter += 1

        data_zs[filename] = {}
        for step in data[filename]:
            data_zs[filename][step] = {}
            struc_dict = data[filename][step]["structure"]
            structure = Structure.from_dict(struc_dict)
            try:
                prediction = chgnet.predict_structure(structure)
                prediction["e"] = prediction["e"].astype(np.float64)
                prediction["f"] = prediction["f"].tolist()
                status = True
            except Exception as e:
                print(f"{filename}-{step} has failed...")
                sys.stdout.flush()
                prediction = {}
                prediction["e"] = 0.0
                prediction["f"] = [[]]
                status = False

            data_zs[filename][step]["structure"] = struc_dict
            data_zs[filename][step]["e_per_at"] = data[filename][step]["e_per_at"]
            data_zs[filename][step]["e_zs"] = prediction["e"]
            data_zs[filename][step]["f_zs"] = prediction["f"]
            data_zs[filename][step]["status"] = status

    print("writing data...")
    sys.stdout.flush()
    write_json(data_zs, fjson)
    return read_json(fjson)

def main():
    data = get_data()
    if data is not None:
        res = run_zero_shot(data, remake=True) 
    else:
        print("Data is none")
    return None

if __name__ == "__main__":
    main()
