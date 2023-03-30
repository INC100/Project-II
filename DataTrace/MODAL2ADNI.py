import os
import pandas as pd
import shutil

# for root, dirs, files in os.walk('../Data'):
#     for file in files:
#         print(file)


#读取ADNI全.xlsx数据，转为np.array形式
data = pd.read_excel('./Data/ADNI全.xlsx')
data = data.values
subject_id = data[:,3]
subject_group = data[:,-2]

#依据subject_id和subject_group，将Data内的数据分为ADNI1和ADNI2
#ADNI1
ADNI1 = []
for i in range(len(subject_id)):
    if subject_group[i] == 'ADNI1':
        ADNI1.append(subject_id[i])
ADNI1 = list(set(ADNI1))
#ADNI2
ADNI2 = []
for i in range(len(subject_id)):
    if subject_group[i] == 'ADNI2':
        ADNI2.append(subject_id[i])
ADNI2 = list(set(ADNI2))

for i in ADNI1:

    i = str(i)

    subject_path = './Data/ADNI1/'+i
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    mrpath = './Data/5.aBEAT_tillStrip/'+i+'/Normalised_'+i+'_acpc_orig_N3Corrected-0-reoriented-strip.mat'
    petpath = './Data/7.PET_ADNI2/'+i+'/Normalised_striped_PET.mat'

    if os.path.exists(mrpath):
        shutil.copy(mrpath,subject_path+'/MRI.mat')
    if os.path.exists(petpath):
        shutil.copy(petpath,subject_path+'/PET.mat')

    print(i,'-ADNI1')


for i in ADNI2:

    i = str(i)

    subject_path = './Data/ADNI2/'+i
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    mrpath = './Data/5.aBEAT_tillStrip/'+i+'/Normalised_'+i+'_acpc_orig_N3Corrected-0-reoriented-strip.mat'
    petpath = './Data/7.PET_ADNI2/'+i+'/Normalised_striped_PET.mat'

    if os.path.exists(mrpath):
        shutil.copy(mrpath,subject_path+'/MRI.mat')
    if os.path.exists(petpath):
        shutil.copy(petpath,subject_path+'/PET.mat')

    print(i,'-ADNI2')