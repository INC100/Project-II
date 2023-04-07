import os
import numpy as np
import scipy.io as sio
import pandas as pd
from DataGenerator import MRPETDATASET, VisualDataset, tensor_to_numpy
from torch.utils.data import DataLoader
from DiscussionModel import Discriminator, Variant_pooling, Variant_Upsampling
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
def tranditional_pooling(mode = 'avgpooling'):

    if mode == 'avgpooling' or mode == 'maxpooling':
        #for avgpooling and maxpooling
        g = Variant_pooling(pooling_operation=mode).to(device)
        filepath = './Discussion/Pooling/'+mode+'/'+trainstage+'/'
        g.load_state_dict(torch.load(filepath + 'TransGenerator_40.pth'))
        imagepath = './imageshow/Pooling/'+mode+'/'+trainstage+'/'
        feature_extractor = torch.nn.Sequential(*list(g.sh_enc.children())[:2])
        ori_feature = feature_extractor(mrimgs)
        pooled_feature = g.sh_enc[2](ori_feature)
        fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)
    elif mode == 'convstride':
        #for convstride
        g = Variant_pooling(pooling_operation=mode).to(device)
        filepath = './Discussion/Pooling/'+mode+'/'+trainstage+'/'
        g.load_state_dict(torch.load(filepath + 'TransGenerator_40.pth'))
        imagepath = './imageshow/Pooling/'+mode+'/'+trainstage+'/'
        feature_extractor = torch.nn.Sequential(*list(g.sh_enc.children())[:1])
        ori_feature = feature_extractor(mrimgs)
        pooled_feature = g.sh_enc[1](ori_feature)
        fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)
    elif mode == 'TAP':
        #for TAP
        g = TransGenerator().to(device)
        filepath = './Models/Fidelity GAN/' + trainstage + '/'
        g.load_state_dict(torch.load(filepath + 'TransGenerator_80.pth'))
        imagepath = './imageshow/Pooling/TAP/' + trainstage + '/'
        feature_extractor = torch.nn.Sequential(*list(g.sh_enc_1.children())[:2])
        ori_feature = feature_extractor(mrimgs)
        pooled_feature, a, b = g.sh_enc_1[2](ori_feature)
        fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1 = g(mrimgs, petimgs)

    if not os.path.exists(imagepath): os.makedirs(imagepath)

    ori_feature = tensor_to_numpy(ori_feature)[0, 6, :, 70, :]
    slice_pooled_feature = tensor_to_numpy(pooled_feature)[0, 6, :, 35, :]

    errormap = torch.nn.L1Loss(reduce=False)(petimgs, fakepet2)
    errormap = tensor_to_numpy(errormap)[0, 0, :, 70, :]

    # 保存ori_feature
    plt.imshow(ori_feature, cmap='gray')
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_orifeature_box.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()


    # 保存pooled_feature
    plt.imshow(slice_pooled_feature, cmap='gray', vmax=max(ori_feature.flatten()))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_pooledfeature_box.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()


    # 画出ori_feature的采样位置
    if mode == 'TAP':
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
        plt.savefig(imagepath + str(subjectid) + '_sampling.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    elif mode == 'convstride' or mode=='avgpooling':
        plt.imshow(errormap, cmap='gray')
        for i in range(0, 144, 2):
            for j in range(0, 176, 2):
                plt.gca().add_patch(plt.Rectangle((i, j), 2, 2, color='red', fill=False,
                                                  linewidth=0.2))
        plt.axis('off')
        plt.savefig(imagepath + str(subjectid) + '_sampling.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    elif mode == 'maxpooling':
        plt.imshow(errormap, cmap='gray')
        #计算2*2的采样窗口内最大值的位置
        for i in range(0, 144, 2):
            for j in range(0, 176, 2):
                window = errormap[j:j+2, i:i+2]
                maxindex1, maxindex0 = np.where(window == np.max(window))
                plt.scatter(i+maxindex1[0], j+maxindex0[0], s=0.2, c='r')
        plt.axis('off')
        plt.savefig(imagepath + str(subjectid) + '_sampling.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    print('subjectid: ', subjectid)

def transposeconv():
    g = Variant_Upsampling(upsampling_operation='deconv').to(device)
    filepath = './Discussion/Upsampling/deconv/' + trainstage + '/'
    g.load_state_dict(torch.load(filepath + 'TransGenerator_75.pth'))
    imagepath = './imageshow/Upsampling/deconv/' + trainstage + '/'
    if not os.path.exists(imagepath): os.makedirs(imagepath)

    fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1, l_features = g(mrimgs, petimgs)
    feature_extractor = torch.nn.Sequential(*list(g.pet_dec.children())[:6])
    l_features = feature_extractor(l_features)
    h_features = g.pet_dec[6:9](l_features)

    max_gray = max(tensor_to_numpy(l_features).flatten())

    # 保存fakepet2
    plt.imshow(tensor_to_numpy(fakepet2)[0, 0, :, 70, :], cmap='gray')
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_fakepet2.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 验证l_feature的哪个位置会被采样到h_feature
    plt.imshow(tensor_to_numpy(l_features)[0, 6, :, 35, :], cmap='gray')
    plt.gca().add_patch(
        plt.Rectangle((44, 36), 2, 2, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_l.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # 在h_feature位置画圈
    plt.imshow(tensor_to_numpy(h_features)[0, 3, :, 70, :], cmap='gray', vmax=max_gray)
    plt.gca().add_patch(plt.Circle((88, 72), 1, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_h.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

def NI_TI(mode='trilinear'):
    # upsampling_operation应该在['pixelshuffle', 'nearest', 'trilinear', 'deconv', 'qkv]
    g = Variant_Upsampling(upsampling_operation=mode).to(device)
    filepath = './Discussion/Upsampling/'+mode+'/' + trainstage + '/'
    g.load_state_dict(torch.load(filepath + 'TransGenerator_90.pth'))
    imagepath = './imageshow/Upsampling/'+mode+'/' + trainstage + '/'
    if not os.path.exists(imagepath): os.makedirs(imagepath)

    fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1, l_features = g(mrimgs, petimgs)
    feature_extractor = torch.nn.Sequential(*list(g.pet_dec.children())[:6])
    l_features = feature_extractor(l_features)
    h_features = g.pet_dec[6](l_features)

    max_gray = max(tensor_to_numpy(l_features).flatten())

    # 保存fakepet2
    plt.imshow(tensor_to_numpy(fakepet2)[0, 0, :, 70, :], cmap='gray')
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_fakepet2.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 验证l_feature的哪个位置会被采样到h_feature
    plt.imshow(tensor_to_numpy(l_features)[0, 0, :, 35, :], cmap='gray', vmax=max_gray)
    plt.gca().add_patch(
        plt.Rectangle((44, 36), 2, 2, color='red', fill=False, linewidth=1))
    # for i in range(-5, 6):
    #     for j in range(-5, 6):
    #         location_matrix = torch.zeros_like(l_features, device=device)
    #         location_matrix[:, :, 44 + i, 36, 36 + j] = 1
    #
    #         fake_h_feature = g.pet_dec[6](location_matrix)
    #         if fake_h_feature[0, 0, 88, 72, 72]>0:
    #             plt.gca().add_patch(plt.Circle((44 + i, 36 + j), 1, color='red', fill=False, linewidth=1))
    #             print('i', i, 'j', j, 'value', fake_h_feature[0, 0, 88, 72, 72].item())
    #
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_l.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # 在h_feature位置画圈
    plt.imshow(tensor_to_numpy(h_features)[0, 0, :, 70, :], cmap='gray', vmax=max_gray)
    plt.gca().add_patch(plt.Circle((88, 72), 1, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_h.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

def pixelshuffle():
    # upsampling_operation应该在['pixelshuffle', 'nearest', 'trilinear', 'deconv', 'qkv]
    g = Variant_Upsampling(upsampling_operation='pixelshuffle').to(device)
    filepath = './Discussion/Upsampling/pixelshuffle/' + trainstage + '/'
    g.load_state_dict(torch.load(filepath + 'TransGenerator_90.pth'))
    imagepath = './imageshow/Upsampling/pixelshuffle/' + trainstage + '/'
    if not os.path.exists(imagepath): os.makedirs(imagepath)

    fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1, l_features = g(mrimgs, petimgs)
    feature_extractor = torch.nn.Sequential(*list(g.pet_dec.children())[:6])
    l_features = feature_extractor(l_features)
    h_features = g.pet_dec[6](l_features)

    max_gray = max(tensor_to_numpy(h_features).flatten())

    # 保存fakepet2
    plt.imshow(tensor_to_numpy(fakepet2)[0, 0, :, 70, :], cmap='gray')
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_fakepet2.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 验证l_feature的哪个位置会被采样到h_feature
    plt.imshow(tensor_to_numpy(l_features)[0, 0, :, 35, :], cmap='gray', vmax=max_gray)
    plt.gca().add_patch(plt.Circle((44, 36), 1, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_l.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # 在h_feature位置画圈
    plt.imshow(tensor_to_numpy(h_features)[0, 0, :, 70, :], cmap='gray', vmax=max_gray)
    # plt.gca().add_patch(plt.Circle((88, 72), 1, color='red', fill=False, linewidth=1))
    plt.gca().add_patch(
        plt.Rectangle((88, 72), 2, 2, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_h.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

def ACE():
    g = TransGenerator().to(device)
    filepath = './Models/Fidelity GAN/' + trainstage + '/'
    g.load_state_dict(torch.load(filepath + 'TransGenerator_120.pth'))
    imagepath = './imageshow/Upsampling/ACE/' + trainstage + '/'
    if not os.path.exists(imagepath): os.makedirs(imagepath)

    fakemrs, fakepet1, fakepet2, L_reg, L_kl, p_d1, p_d2, m_d1, m_d2, m_c1, mr2pet = g(mrimgs, petimgs)
    feature_extractor = torch.nn.Sequential(*list(g.pet_dec.children())[:6])
    l_features = feature_extractor(mr2pet)
    h_features = g.pet_dec[6](l_features)

    # 保存fakepet2
    plt.imshow(tensor_to_numpy(fakepet2)[0, 0, :, 70, :], cmap='gray')
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_fakepet2.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 验证l_feature的哪个位置会被采样到h_feature
    plt.imshow(tensor_to_numpy(l_features)[0, 3, :, 35, :], cmap='gray')
    for i in range(43, 49):
        for j in range(30, 41):
            location_matrix = torch.zeros_like(l_features, device=device)
            location_matrix[:, :, i, 35, j] = 1
            fake_h = g.pet_dec[6](location_matrix)
            if fake_h[0, 0, 88, 70, 72] > 0.1:
                print(i, j, fake_h[0, 0, 88, 70, 72])
                # plt.gca().add_patch(plt.Circle((j, i), 1, color='red', fill=False, linewidth=1))
                plt.scatter(j, i, c='r', s=1)
            else:
                print(i, j)
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_l.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # 在h_feature位置画圈
    plt.imshow(tensor_to_numpy(h_features)[0, 0, :, 70, :], cmap='gray')
    plt.gca().add_patch(plt.Circle((72, 88), 1, color='red', fill=False, linewidth=1))
    plt.axis('off')
    plt.savefig(imagepath + str(subjectid) + '_h.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()


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
        # tranditional_pooling(mode='TAP')


        break
