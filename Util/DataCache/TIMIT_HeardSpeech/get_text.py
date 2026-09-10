import pickle
import numpy as np
import torch.nn.functional as F 


from scipy.io import loadmat
from pathlib import Path

def mat_to_data_dict(
    mat         : dict,
    audio_dir   : Path
) -> dict:

    all_data = {}

    mat_indices = mat['evt_ECI_TCPIP_55513'][0]
    eeg_indices = mat['evt_ECI_TCPIP_55513'][3]
    count       = 0

    print(mat_indices)

    for start in range(0, len(mat_indices), 2):
        end = start + 1

        try:
            start_item_mat, end_item_mat, start_item_eeg, end_item_eeg = (
                mat_indices[start].item(),
                mat_indices[end].item(),
                int(eeg_indices[start].item()),
                int(eeg_indices[end].item())
            )
            
            if start_item_mat == 'SBLS' or start_item_mat == 'EBLS': continue
            if end_item_mat   == 'EBLS' or end_item_mat   == 'EBLE': continue
    
            speaker_id = start_item_mat[1]
            audio_id   = start_item_mat[2]
    
            for key, item in mat.items():
                if isinstance(item, np.ndarray):
                    if item.ndim == 2 and item.shape[0] == 129:
                        eeg_data = item[:-1, start_item_eeg:end_item_eeg]

                        with open(audio_dir / f"sa{speaker_id}.{audio_id}.lab", "r") as f:
                            for line in f:
                                line = line.strip()
                                if not line: continue

                                

                        all_data[f"{count:04d}_{audio_id}_{speaker_id}"] = eeg_data
                        count += 1 

        except Exception as e:
            print(e)
            continue
    
    return all_data
        

if __name__ == '__main__':
    root    = Path("Data/TIMIT_HEARONLY/Timit_Hear_Only_Sa1-Sa2")
    out_dir = Path("Data/TIMIT_HEARONLY/TIMIT_EXTRACTED_25SYLL")
    out_dir.mkdir(exist_ok=True, parents=True)

    count_lookup    = {}
    matched_mats    = sorted(path for path in root.rglob("*_0_60_notch50*.mat") if path.is_file())

    for mat in matched_mats:
        subject_id  = str(mat).split("/")[-1].split("_")[0]

        if subject_id not in count_lookup.keys(): count_lookup[subject_id] = 0
        else    : count_lookup[subject_id] += 1

        data = mat_to_data_dict(loadmat(mat))

        with open(out_dir / f"{subject_id}_S{count_lookup[subject_id]}.pkl", "wb") as f:
            pickle.dump(data, f)


