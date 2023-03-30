import glob
import re
from torch.utils.data import Dataset
import numpy as np
from scipy import io as sio
from torch.utils.data import DataLoader
import os
import shutil


class MRPETDATASET(Dataset):
    def __init__(self, ADNI = 'ADNI1'):
        self.train_stage = ADNI
        self.pet_path = glob.glob('./Data/' + self.train_stage + '/*/PET.mat')

    def __len__(self):
        return len(self.pet_path)

    def __getitem__(self, item):
        pet_path = self.pet_path[item]
        l_slash_idx = [m.start() for m in re.finditer('/', pet_path)]
        subject_id = pet_path[l_slash_idx[-2] + 1: l_slash_idx[-1]]

        mr_path = pet_path.replace('PET', 'MRI')
        if not os.path.exists(mr_path):
            shutil.copy('./Data/5.aBEAT_tillStrip/'+subject_id+'/Normalised_'+subject_id+'_acpc_orig_N3Corrected-0-reoriented-strip.mat', mr_path)

        pet_img = np.expand_dims(np.nan_to_num(np.swapaxes(sio.loadmat(pet_path)['img'], 0, 1,)) * 255.0, axis = 0).astype(np.float32)
        mr_img = np.expand_dims(np.nan_to_num(np.swapaxes(sio.loadmat(mr_path)['img'], 0, 1,)) * 255.0, axis = 0).astype(np.float32)


        return subject_id, mr_img, pet_img
    

def tensor_to_numpy(x):
    return x.cpu().detach().numpy()
