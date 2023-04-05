import os
import numpy as np
import scipy.io as sio
import pandas as pd
from DataGenerator import MRPETDATASET, VisualDataset, tensor_to_numpy
from torch.utils.data import DataLoader
from DiscussionModel import Discriminator, Variant_pooling
from ModelConstruct import TransGenerator
from skimage.measure import compare_ssim as ssim_fn
from skimage.measure import compare_psnr as psnr_fn
import torch
import matplotlib.pyplot as plt
import tqdm

def metrics_assess(subjectid, mrimgs, petimgs):
    fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)
    mae_pet2 = torch.nn.L1Loss(size_average=True)(petimgs, fakepet2)

    mae_pet2 = tensor_to_numpy(mae_pet2)
    petimgs = tensor_to_numpy(petimgs)
    fakepet2 = tensor_to_numpy(fakepet2)
    subjectid = int(subjectid[0])

    train_p2_ssim = ssim_fn(fakepet2[0, 0, :, :, :], petimgs[0, 0, :, :, :],
                            multichannel=True, data_range=255.0)
    train_p2_psnr = psnr_fn(petimgs[0, 0, :, :, :], fakepet2[0, 0, :, :, :],
                            data_range=255.0)

    print('subjectid: ', subjectid, 'mae_pet2: ', mae_pet2, 'train_p2_ssim: ', train_p2_ssim, 'train_p2_psnr: ', train_p2_psnr)
def epoch_field(epoch):

    feature_extractor = torch.nn.Sequential(*list(g.sh_enc_1.children())[:2])
    ori_feature = feature_extractor(mrimgs)
    pooled_feature, a, b = g.sh_enc_1[2](ori_feature)
    fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1 = g(mrimgs, petimgs)


    ori_feature = tensor_to_numpy(ori_feature)[0, 6, :, 70, :]
    slice_pooled_feature = tensor_to_numpy(pooled_feature)[0, 6, :, 35, :]

    errormap = torch.nn.L1Loss(reduce=False)(petimgs, fakepet2)
    errormap = tensor_to_numpy(errormap)[0, 0, :, 70, :]

    c1 = tensor_to_numpy(m_c1)
    c1[:, :, :, :, 0] = c1[:, :, :, :, 0] * 88 + 88
    c1[:, :, :, :, 1] = c1[:, :, :, :, 1] * 72 + 72
    c1[:, :, :, :, 2] = c1[:, :, :, :, 2] * 72 + 72
    c1 = np.array(c1, dtype=np.int)

    c1_coor = np.where(c1[0, :, :, :, 1] == 70)
    c1_sampling = c1[0][c1_coor]

    plt.imshow(errormap, cmap='gray')
    plt.axis('off')
    plt.scatter(c1_sampling[:,2], c1_sampling[:,0], s=0.2, c='r')
    plt.savefig(imagepath + str(subjectid) + '_sampling_'+str(epoch)+'.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    print('subjectid: ', subjectid)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda:0'

stage = 'ADNI1'
trainstage, teststage = ('ADNI1', 'ADNI2') if stage == 'ADNI1' else ('ADNI2', 'ADNI1')
# testdataset = DataLoader(MRPETDATASET(ADNI=teststage), batch_size=1, shuffle=False, drop_last= False)
testdataset = DataLoader(VisualDataset(ADNI=teststage,
                                       subjectid=[538, 661, 704, 940, 646, 466, 667, 494]),
                         batch_size=1, shuffle=False, drop_last= False)

with torch.no_grad():
    for n_batch, (subjectid, mrimgs, petimgs) in enumerate(testdataset):
        mrimgs = mrimgs.to(device); petimgs = petimgs.to(device); subjectid = int(subjectid[0])

        #mode==['maxpooling', 'avgpooling','convstride', 'TAP']中
        g = TransGenerator().to(device)
        filepath = './Models/Fidelity GAN/' + trainstage + '/'
        imagepath = './imageshow/epoch_field/' + trainstage + '/'
        if not os.path.exists(imagepath): os.makedirs(imagepath)

        for epoch in [5, 50, 100]:
            g.load_state_dict(torch.load(filepath + 'TransGenerator_' + str(epoch) + '.pth'))
            epoch_field(epoch)
        break
