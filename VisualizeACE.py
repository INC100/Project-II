import os

import matplotlib.pyplot as plt
import numpy as np
from DataGenerator import MRPETDATASET, VisualDataset, tensor_to_numpy
from torch.utils.data import DataLoader
from ModelConstruct import TransGenerator
from sklearn.cluster import KMeans
import torch
import scipy.io as sio

import matplotlib.colors as mcolors
colors=list(mcolors.CSS4_COLORS.values())

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

        g = TransGenerator().to(device)
        filepath = './Models/Fidelity GAN/' + trainstage + '/'
        g.load_state_dict(torch.load(filepath + 'TransGenerator_155.pth'))
        imagepath = './imageshow/Upsampling/ACE/' + trainstage + '/'
        if not os.path.exists(imagepath): os.makedirs(imagepath)

        mr_feature, mr_d1, mr_c1 = g.sh_enc_1(mrimgs)
        mr_feature, mr_d2, mr_c2 = g.sh_enc_2(mr_feature)
        feature_mr_1 = g.sh_rsb1(mr_feature)
        feature_mr_2 = g.sh_rsb2(feature_mr_1)
        feature_mr_3 = g.sh_rsb3(feature_mr_2)
        rmr_to_pet1, p_mean1, p_std1 = g.tran_1(feature_mr_1)
        mr_to_pet1 = g.sh_rsb2(rmr_to_pet1)
        mr_to_pet1 = g.sh_rsb3(mr_to_pet1)
        rmr_to_pet2, p_mean2, p_std2 = g.tran_2(feature_mr_2)
        mr_to_pet2 = g.sh_rsb3(rmr_to_pet2)
        rmr_to_pet3, p_mean3, p_std3 = g.tran_3(feature_mr_3)
        mr2pet = (mr_to_pet1 + mr_to_pet2 + rmr_to_pet3) / 3.0

        feature_extractor = torch.nn.Sequential(*list(g.pet_dec.children())[:4])
        l_features = feature_extractor(mr2pet)
        l_features, semantic_matrix = g.pet_dec[4](l_features)
        h_features, semantic_matrix = torch.nn.Sequential(*list(g.pet_dec.children())[5:7])(l_features)

        slice = 70

        semantic_matrix = torch.swapaxes(semantic_matrix, 1, 4)
        semantic_matrix = tensor_to_numpy(semantic_matrix)[0, :, slice, :, :, 0]
        semantic_matrix = np.reshape(semantic_matrix, (semantic_matrix.shape[0], -1))
        semantic_matrix = np.transpose(semantic_matrix)

        template = sio.loadmat('./DataTrace/538_aal.mat')['img'][slice, :, :]

        nclusters = 0
        for k in np.unique(template):
            if np.sum(template == k)< 100:
                template[template == k] = 0
            else:
                nclusters += 1

        #使用sci-kit learn中的kmeans聚类将semantic_matrix聚类为90类
        kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(semantic_matrix)
        labels = kmeans.labels_
        labels = np.reshape(labels, (144, 144))

        plt.figure(1)
        plt.imshow(tensor_to_numpy(h_features[0, 0, slice, :, :]), cmap='gray')
        plt.figure(2)
        plt.imshow(tensor_to_numpy(h_features[0, 0, slice, :, :]), cmap='gray')
        for i in np.unique(template)[1:]:
            x_location, y_location = np.where((template == i))

            for _ in range(len(x_location)):
                plt.figure(1)
                plt.scatter(y_location[_], x_location[_], c=colors[labels[x_location[_], y_location[_]]], s=1)
                plt.figure(2)
                plt.scatter(y_location[_], x_location[_], c=colors[i], s=1, alpha=1)
            print(i)

        plt.figure(1)
        plt.axis('off')
        plt.savefig(imagepath + str(subjectid) +'semantic.png', bbox_inches='tight')
        plt.close()

        plt.figure(2)
        plt.axis('off')
        plt.savefig(imagepath + str(subjectid) +'template.png', bbox_inches='tight')
        plt.close()


        break
