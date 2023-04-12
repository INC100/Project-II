import os
import numpy as np
import pandas as pd
from DataGenerator import MRPETDATASET, tensor_to_numpy
from torch.utils.data import DataLoader
from DiscussionModel import Discriminator, Variant_Direct
from skimage.measure import compare_ssim as ssim_fn
from skimage.measure import compare_psnr as psnr_fn
import torch
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

device = 'cuda:0'

stage = 'ADNI1'
trainstage, teststage = ('ADNI1', 'ADNI2') if stage == 'ADNI1' else ('ADNI2', 'ADNI1')
traindataset = DataLoader(MRPETDATASET(ADNI=trainstage), batch_size=1, shuffle=True, drop_last= False)
testdataset = DataLoader(MRPETDATASET(ADNI=teststage), batch_size=1, shuffle=True, drop_last= False)

g = Variant_Direct().to(device)
p_d = Discriminator().to(device)
filepath = './Discussion/Direct/'+trainstage+'/'
if not os.path.exists(filepath): os.makedirs(filepath)

g_op = torch.optim.Adam(lr= 0.0002, params= g.parameters(), eps= 1e-5)
pd_op = torch.optim.Adam(lr= 0.0002, params= p_d.parameters(), eps= 1e-5)

"""
损失权重
"""
lamd = 0.01

epoches = 200
for epoch in range(1, epoches + 1):
    trainpbar = tqdm.tqdm(total= len(traindataset))
    train_mr_mae = 0; train_mr_ssim = 0; train_mr_psnr = 0;
    train_pet_mae = 0; train_pet_ssim = 0; train_pet_psnr = 0;
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
        fakepet, m_d1, m_d2, m_c1 = g(mrimgs)

        with torch.no_grad():
            errormap = torch.nn.L1Loss(size_average=False, reduce=False)(fakepet, petimgs)
            errormap_d1 = torch.nn.AvgPool3d(2)(errormap)
            errormap_c1 = torch.nn.functional.grid_sample(input = errormap, grid = m_c1)
            errormap_d2 = torch.nn.AvgPool3d(2)(errormap_c1)


        dis_fakepet = p_d(fakepet)
        adv_pet = torch.nn.MSELoss(size_average=True)(realtarget, dis_fakepet)

        mae_pet = torch.nn.L1Loss(size_average=True)(petimgs, fakepet)

        mr_d1_loss = torch.nn.L1Loss(size_average=True, reduction='mean')(errormap_d1, m_d1)
        mr_d2_loss = torch.nn.L1Loss(size_average=True, reduction='mean')(errormap_d2, m_d2)
        L_d = mr_d1_loss + mr_d2_loss

        g_loss = adv_pet + mae_pet + L_d * lamd
        g_loss.backward()
        g_op.step()


        ##########################################训练Discriminator##########################################
        pd_op.zero_grad()
        with torch.no_grad():
            fakepet, m_d1, m_d2, m_c1 = g(mrimgs)
        dis_realpet = p_d(petimgs)
        dis_fakepet = p_d(fakepet)
        pr_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_realpet)
        pf1_loss = torch.nn.MSELoss(size_average=True)(faketarget, dis_fakepet)
        d_loss = pr_loss + pf1_loss
        d_loss.backward()
        pd_op.step()

        ##########################################计算Metrics##########################################
        petimgs = tensor_to_numpy(petimgs)
        fakepet = tensor_to_numpy(fakepet)
        train_pet_mae = tensor_to_numpy(mae_pet) * batch + train_pet_mae
        for _ in range(batch):
            train_pet_ssim = ssim_fn(fakepet[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + train_pet_ssim
            train_pet_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet[_, 0, :, :, :], data_range=255.0) + train_pet_psnr
        trainpbar.set_postfix({
            'Lg': '{0:1.3f}'.format(tensor_to_numpy(g_loss)),
            '1m': '{0:1.3f}'.format(train_pet_mae / train_nums),
            '1p': '{0:1.3f}'.format(train_pet_psnr / train_nums),
            '1s': '{0:1.3f}'.format(train_pet_ssim / train_nums),
        })
        trainpbar.update(1)
    trainpbar.close()


    if epoch % 5 ==0:
        torch.save(g.state_dict(), filepath + 'TransGenerator_' + str(epoch) + '.pth')
        torch.save(p_d.state_dict(), filepath + 'PetDiscriminator_' + str(epoch) + '.pth')
        testpbar = tqdm.tqdm(total= len(testdataset))
        test_pet_mae = 0; test_pet_ssim = 0; test_pet_psnr = 0;
        test_nums = 0;
        for n_batch, (subjectid, mrimgs, petimgs) in enumerate(testdataset):
            mrimgs = mrimgs.to(device); petimgs = petimgs.to(device)
            testpbar.set_description('[Testing {0}/{1}] | [Batch {2}/{3}]'.format(epoch, epoches, n_batch + 1, len(testdataset)))
            batch, channel, height, width, depth = mrimgs.shape
            test_nums = test_nums + batch
            ##########################################计算Metrics##########################################
            with torch.no_grad():
                fakepet, m_d1, m_d2, m_c1 = g(mrimgs)
                mae_pet = torch.nn.L1Loss(size_average=True)(petimgs, fakepet)
            test_pet_mae = tensor_to_numpy(mae_pet) * batch + test_pet_mae
            petimgs = tensor_to_numpy(petimgs)
            fakepet = tensor_to_numpy(fakepet)
            for _ in range(batch):
                test_pet_ssim = ssim_fn(fakepet[_, 0, :, :, :], petimgs[_, 0, :, :, :], multichannel=True, data_range=255.0) + test_pet_ssim
                test_pet_psnr = psnr_fn(petimgs[_, 0, :, :, :], fakepet[_, 0, :, :, :], data_range=255.0) + test_pet_psnr
            testpbar.set_postfix({
                '1m': '{0:1.3f}'.format(test_pet_mae / test_nums),
                '1p': '{0:1.3f}'.format(test_pet_psnr / test_nums),
                '1s': '{0:1.3f}'.format(test_pet_ssim / test_nums)
            })
            testpbar.update(1)
        testpbar.close()

        #########################################保存模型和结果##########################################
        if not os.path.exists(filepath + 'records.xlsx'):
            title = np.array([['epoch',
                               'train_pet_mae', 'train_pet_ssim', 'train_pet_psnr',
                               'test_pet_mae', 'test_pet_ssim', 'test_pet_psnr',
                               ]])
            data = np.concatenate((title, np.expand_dims(np.array([
                epoch,
                train_pet_mae / train_nums, train_pet_ssim / train_nums, train_pet_psnr / train_nums,
                test_pet_mae / test_nums, test_pet_ssim / test_nums, test_pet_psnr / test_nums,
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
                train_pet_mae / train_nums, train_pet_ssim / train_nums, train_pet_psnr / train_nums,
                test_pet_mae / test_nums, test_pet_ssim / test_nums, test_pet_psnr / test_nums,
                ]), axis=0)), axis=0))
            writer = pd.ExcelWriter(filepath + 'records.xlsx')
            new_data.to_excel(writer, header=False)
            writer.save()
