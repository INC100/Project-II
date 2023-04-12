import os
from DataGenerator import MRPETDATASET, tensor_to_numpy
from torch.utils.data import DataLoader
from ModelConstruct import TransGenerator
from DiscussionModel import Variant_Upsampling

import torch
import tqdm
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda:0'

stage = 'ADNI1'
trainstage, teststage = ('ADNI1', 'ADNI2') if stage == 'ADNI1' else ('ADNI2', 'ADNI1')
testdataset = DataLoader(MRPETDATASET(ADNI=teststage), batch_size=1, shuffle=True, drop_last= False)

g = TransGenerator().to(device)
g = Variant_Upsampling(upsampling_operation='pixelshuffle').to(device)

# filepath = './Models/Fidelity GAN/'+trainstage+'/'
modelfilepath = './Discussion/Upsampling/pixelshuffle/'+trainstage+'/'
g.load_state_dict(torch.load(modelfilepath + 'TransGenerator_190.pth'))

filepath = './Data/Completion/F-GAN/' + teststage + '/'

testpbar = tqdm.tqdm(total= len(testdataset))
for n_batch, (subjectid, mrimgs, petimgs) in enumerate(testdataset):
    mrimgs = mrimgs.to(device); petimgs = petimgs.to(device)
    testpbar.set_description('[[Batch {0}/{1}]'.format(n_batch, len(testdataset)))
    batch, channel, height, width, depth = mrimgs.shape
    ##########################################计算Metrics##########################################
    with torch.no_grad():
        fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1 = g(mrimgs, petimgs)
    fakepet2 = tensor_to_numpy(fakepet2)/255.0
    if not os.path.exists(filepath + subjectid[0]): os.makedirs(filepath + subjectid[0])
    sio.savemat(filepath + subjectid[0] + '/PET.mat', {'img': fakepet2})
    testpbar.update(1)

testpbar.close()

