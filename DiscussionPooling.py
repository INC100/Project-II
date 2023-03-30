import os
import numpy as np
import scipy.io as sio
import pandas as pd
from DataGenerator import MRPETDATASET, tensor_to_numpy
from torch.utils.data import DataLoader
from DiscussionModel import Discriminator, Variant_pooling
from skimage.measure import compare_ssim as ssim_fn
from skimage.measure import compare_psnr as psnr_fn
import torch
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device = 'cuda:0'

matrix_size = 1
cuda_matrix = torch.tensor(torch.rand(size=(matrix_size, matrix_size, matrix_size),device=device))

stage = 'ADNI1'
trainstage, teststage = ('ADNI1', 'ADNI2') if stage == 'ADNI1' else ('ADNI2', 'ADNI1')
traindataset = DataLoader(MRPETDATASET(ADNI=trainstage), batch_size=1, shuffle=True, drop_last= False)
testdataset = DataLoader(MRPETDATASET(ADNI=teststage), batch_size=1, shuffle=True, drop_last= False)

g = Variant_pooling(pooling_operation='maxpooling').to(device) #pooling_operation应在['maxpooling', 'avgpooling','convstride']
m_d = Discriminator().to(device)
p_d = Discriminator().to(device)

filepath = './Discussion/Pooling/maxpooling/'+trainstage+'/'
if not os.path.exists(filepath): os.makedirs(filepath)

g_op = torch.optim.Adam(lr= 0.0002, params= g.parameters(), eps= 1e-5)
md_op = torch.optim.Adam(lr= 0.0002, params= m_d.parameters(), eps= 1e-5)
pd_op = torch.optim.Adam(lr= 0.0002, params= p_d.parameters(), eps= 1e-5)

"""
损失权重
"""
lama = lamkl = lamreg = lamg =1;
lamp2 = 1; lamp1 = lamm = 0.5

epoches = 200
for epoch in range(1, epoches + 1):
    trainpbar = tqdm.tqdm(total= len(traindataset))
    train_mr_mae = 0; train_mr_ssim = 0; train_mr_psnr = 0;
    train_p1_mae = 0; train_p1_ssim = 0; train_p1_psnr = 0;
    train_p2_mae = 0; train_p2_ssim = 0; train_p2_psnr = 0;
    train_nums = 0;
    for n_batch, (subjectid, mrimgs, petimgs) in enumerate(traindataset):
        mrimgs = mrimgs.to(device); petimgs = petimgs.to(device)
        trainpbar.set_description('[Epoch {0}/{1}] | [Batch {2}/{3}]'.format(epoch, epoches, n_batch + 1, len(traindataset)))
        batch, channel, height, width, depth = mrimgs.shape
        realtarget = torch.cat((torch.zeros(size=(batch, 1)), torch.ones(size=(batch, 1)))).to(device)
        faketarget = torch.cat((torch.ones(size=(batch, 1)), torch.zeros(size=(batch, 1)))).to(device)
        losstarget = torch.zeros(size=(batch,)).to(device)
        train_nums = train_nums + batch
        ##########################################训练Generator##########################################
        g_op.zero_grad()
        fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)

        dis_fakemrs = m_d(fakemrs)
        dis_fakepet1 = p_d(fakepet1)
        dis_fakepet2 = p_d(fakepet2)
        adv_mr = torch.nn.MSELoss(size_average=True)(realtarget, dis_fakemrs)
        adv_pet1 = torch.nn.MSELoss(size_average=True)(realtarget, dis_fakepet1)
        adv_pet2 = torch.nn.MSELoss(size_average=True)(realtarget, dis_fakepet2)
        L_adv = adv_mr * lamm + adv_pet1 * lamp1 + adv_pet2 * lamp2

        mae_mr = torch.nn.L1Loss(size_average=True)(mrimgs, fakemrs)
        mae_pet1 = torch.nn.L1Loss(size_average=True)(petimgs, fakepet1)
        mae_pet2 = torch.nn.L1Loss(size_average=True)(petimgs, fakepet2)
        L_g = mae_mr * lamm + mae_pet1 * lamp1 + mae_pet2 * lamp2

        g_loss = L_adv * lama + L_kl * lamkl + L_reg * lamreg + L_g * lamg
        g_loss.backward()
        g_op.step()


        ##########################################训练Discriminator##########################################
        md_op.zero_grad()
        pd_op.zero_grad()
        with torch.no_grad():
            fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)
        dis_realmrs = m_d(mrimgs)
        dis_fakemrs = m_d(fakemrs)
        dis_realpet = p_d(petimgs)
        dis_fakepet1 = p_d(fakepet1)
        dis_fakepet2 = p_d(fakepet2)
        mr_loss = torch.nn.MSELoss(size_average=True)(realtarget, dis_realmrs)
        pr_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_realpet)
        mf_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_fakemrs)
        pf1_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_fakepet1)
        pf2_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_fakepet2)
        d_loss = mr_loss + mf_loss + pr_loss + pf1_loss + pf2_loss
        d_loss.backward()
        md_op.step()
        pd_op.step()

        ##########################################计算Metrics##########################################
        mrimgs = tensor_to_numpy(mrimgs)
        petimgs = tensor_to_numpy(petimgs)
        fakemrs = tensor_to_numpy(fakemrs)
        fakepet1 = tensor_to_numpy(fakepet1)
        fakepet2 = tensor_to_numpy(fakepet2)
        train_mr_mae = tensor_to_numpy(mae_mr) * batch + train_mr_mae
        train_p1_mae = tensor_to_numpy(mae_pet1) * batch + train_p1_mae
        train_p2_mae = tensor_to_numpy(mae_pet2) * batch + train_p2_mae
        for _ in range(batch):
            train_mr_ssim = ssim_fn(fakemrs[_, 0, :, :, :], mrimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + train_mr_ssim
            train_mr_psnr = psnr_fn(mrimgs[_, 0, :, :, :], fakemrs[_, 0, :, :, :], data_range=255.0) + train_mr_psnr
            train_p1_ssim = ssim_fn(fakepet1[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + train_p1_ssim
            train_p1_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet1[_, 0, :, :, :], data_range=255.0) + train_p1_psnr
            train_p2_ssim = ssim_fn(fakepet2[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + train_p2_ssim
            train_p2_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet2[_, 0, :, :, :], data_range=255.0) + train_p2_psnr
        trainpbar.set_postfix({
            'Lg': '{0:1.3f}'.format(tensor_to_numpy(g_loss)),
            'mm': '{0:1.3f}'.format(train_mr_mae / train_nums),
            'mp': '{0:1.3f}'.format(train_mr_psnr / train_nums),
            'ms': '{0:1.3f}'.format(train_mr_ssim / train_nums),
            '1m': '{0:1.3f}'.format(train_p1_mae / train_nums),
            '1p': '{0:1.3f}'.format(train_p1_psnr / train_nums),
            '1s': '{0:1.3f}'.format(train_p1_ssim / train_nums),
            '2m': '{0:1.3f}'.format(train_p2_mae / train_nums),
            '2p': '{0:1.3f}'.format(train_p2_psnr / train_nums),
            '2s': '{0:1.3f}'.format(train_p2_ssim / train_nums)
        })
        trainpbar.update(1)
    trainpbar.close()

    if epoch % 5 ==0:
        testpbar = tqdm.tqdm(total= len(testdataset))
        test_mr_mae = 0; test_mr_ssim = 0; test_mr_psnr = 0;
        test_p1_mae = 0; test_p1_ssim = 0; test_p1_psnr = 0;
        test_p2_mae = 0; test_p2_ssim = 0; test_p2_psnr = 0;
        test_nums = 0;
        for n_batch, (subjectid, mrimgs, petimgs) in enumerate(testdataset):
            mrimgs = mrimgs.to(device); petimgs = petimgs.to(device)
            testpbar.set_description('[Testing {0}/{1}] | [Batch {2}/{3}]'.format(epoch, epoches, n_batch + 1, len(testdataset)))
            batch, channel, height, width, depth = mrimgs.shape
            test_nums = test_nums + batch
            ##########################################计算Metrics##########################################
            with torch.no_grad():
                fakemrs, fakepet1, fakepet2, L_reg, L_kl = g(mrimgs, petimgs)
                mae_mr = torch.nn.L1Loss(size_average=True)(mrimgs, fakemrs)
                mae_pet1 = torch.nn.L1Loss(size_average=True)(petimgs, fakepet1)
                mae_pet2 = torch.nn.L1Loss(size_average=True)(petimgs, fakepet2)
            test_mr_mae = tensor_to_numpy(mae_mr) * batch + test_mr_mae
            test_p1_mae = tensor_to_numpy(mae_pet1) * batch + test_p1_mae
            test_p2_mae = tensor_to_numpy(mae_pet2) * batch + test_p2_mae
            mrimgs = tensor_to_numpy(mrimgs)
            petimgs = tensor_to_numpy(petimgs)
            fakemrs = tensor_to_numpy(fakemrs)
            fakepet1 = tensor_to_numpy(fakepet1)
            fakepet2 = tensor_to_numpy(fakepet2)
            for _ in range(batch):
                test_mr_ssim = ssim_fn(fakemrs[_, 0, :, :, :], mrimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + test_mr_ssim
                test_mr_psnr = psnr_fn(mrimgs[_, 0, :, :, :], fakemrs[_, 0, :, :, :], data_range=255.0) + test_mr_psnr
                test_p1_ssim = ssim_fn(fakepet1[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + test_p1_ssim
                test_p1_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet1[_, 0, :, :, :], data_range=255.0) + test_p1_psnr
                test_p2_ssim = ssim_fn(fakepet2[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + test_p2_ssim
                test_p2_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet2[_, 0, :, :, :], data_range=255.0) + test_p2_psnr
            testpbar.set_postfix({
                'mm': '{0:1.3f}'.format(test_mr_mae / test_nums),
                'mp': '{0:1.3f}'.format(test_mr_psnr / test_nums),
                'ms': '{0:1.3f}'.format(test_mr_ssim / test_nums),
                '1m': '{0:1.3f}'.format(test_p1_mae / test_nums),
                '1p': '{0:1.3f}'.format(test_p1_psnr / test_nums),
                '1s': '{0:1.3f}'.format(test_p1_ssim / test_nums),
                '2m': '{0:1.3f}'.format(test_p2_mae / test_nums),
                '2p': '{0:1.3f}'.format(test_p2_psnr / test_nums),
                '2s': '{0:1.3f}'.format(test_p2_ssim / test_nums)
            })
            testpbar.update(1)
        testpbar.close()

        #########################################保存模型和结果##########################################
        torch.save(g.state_dict(), filepath + 'TransGenerator_'+str(epoch)+'.pth')
        torch.save(m_d.state_dict(), filepath + 'MrDiscriminator_'+str(epoch)+'.pth')
        torch.save(p_d.state_dict(), filepath + 'PetDiscriminator_'+str(epoch)+'.pth')
        if not os.path.exists(filepath + 'records.xlsx'):
            title = np.array([['epoch', 'train_mr_mae', 'train_mr_ssim', 'train_mr_psnr',
                               'train_pet1_mae', 'train_pet1_ssim', 'train_pet1_psnr',
                               'train_pet2_mae', 'train_pet2_ssim', 'train_pet2_psnr',
                               'test_mr_mae', 'test_mr_ssim', 'test_mr_psnr',
                               'test_pet1_mae', 'test_pet1_ssim', 'test_pet1_psnr',
                               'test_pet2_mae', 'test_pet2_ssim', 'test_pet2_psnr'
                               ]])
            data = np.concatenate((title, np.expand_dims(np.array([
                epoch,
                train_mr_mae / train_nums, train_mr_ssim / train_nums, train_mr_psnr / train_nums,
                train_p1_mae / train_nums, train_p1_ssim / train_nums, train_p1_psnr / train_nums,
                train_p2_mae / train_nums, train_p2_ssim / train_nums, train_p2_psnr / train_nums,
                test_mr_mae / test_nums, test_mr_ssim / test_nums, test_mr_psnr / test_nums,
                test_p1_mae / test_nums, test_p1_ssim / test_nums, test_p1_psnr / test_nums,
                test_p2_mae / test_nums, test_p2_ssim / test_nums, test_p2_psnr / test_nums
            ]), axis=0)), axis=0)
            data_tf = pd.DataFrame(data)
            writer = pd.ExcelWriter(filepath + 'records.xlsx')
            data_tf.to_excel(writer, header=False)
            writer.save()
        else:
            ori_data = pd.read_excel(filepath + 'records.xlsx', header=None).values
            new_data = pd.DataFrame(
                np.concatenate((ori_data[:, 1:], np.expand_dims(np.array([
                epoch,
                train_mr_mae / train_nums, train_mr_ssim / train_nums, train_mr_psnr / train_nums,
                train_p1_mae / train_nums, train_p1_ssim / train_nums, train_p1_psnr / train_nums,
                train_p2_mae / train_nums, train_p2_ssim / train_nums, train_p2_psnr / train_nums,
                test_mr_mae / test_nums, test_mr_ssim / test_nums, test_mr_psnr / test_nums,
                test_p1_mae / test_nums, test_p1_ssim / test_nums, test_p1_psnr / test_nums,
                test_p2_mae / test_nums, test_p2_ssim / test_nums, test_p2_psnr / test_nums
                ]), axis=0)), axis=0))
            writer = pd.ExcelWriter(filepath + 'records.xlsx')
            new_data.to_excel(writer, header=False)
            writer.save()
