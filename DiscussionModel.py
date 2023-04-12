import warnings
import numpy as np
import torch
from torch.nn.functional import kl_div
import torch.nn.functional as F

warnings.filterwarnings("ignore")
image_shape = (176, 144, 144, 1)
Generator_filters = 1
Discriminator_filters = 1

class PixelShuffle3d(torch.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class skiplayer(torch.nn.Module):
    def __init__(self, infilters):
        super(skiplayer, self).__init__()
        self.layer_1 = self.cksn(infilters=infilters, outfilters=infilters, k_size=3, activation=True, name='layer_1')
        self.layer_2 = self.cksn(infilters=infilters, outfilters=infilters, k_size=3, activation=False, name='layer_2')

    def cksn(input, infilters, outfilters, k_size, activation=True, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D',
                          module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters, kernel_size=k_size,
                                                 stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        if activation: module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        z = torch.add(x, y)
        return z

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = self.cksn(infilters=1, outfilters=4 * Generator_filters, k_size=4, k_stride=2, name='layer_1')
        self.layer_2 = self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=4,
                                 k_stride=2, name='layer_2')
        self.layer_3 = self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=4,
                                 k_stride=2, name='layer_3')
        self.layer_4 = self.cksn(infilters=16 * Generator_filters, outfilters=32 * Generator_filters, k_size=4,
                                 k_stride=2, name='layer_4')
        self.layer_5 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=32 * Generator_filters, out_channels=1, kernel_size=4, stride=1,
                            padding='same'),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=891, out_features=2),
            torch.nn.Softmax()
        )

    def cksn(input, infilters, outfilters, k_size, k_stride, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D',
                          module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters, kernel_size=k_size,
                                                 stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        if k_stride > 1: module.add_module(name=name + 'pooling',
                                           module=torch.nn.AvgPool3d(kernel_size=k_stride, stride=k_stride))
        return module

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        return x

class QKV_Upsampling(torch.nn.Module):
    def __init__(self, filters, upscale, neighbor):
        super(QKV_Upsampling, self).__init__()
        self.kernelsize = 2
        self.q_layer = torch.nn.Sequential(
            self.cksn(infilters=filters, outfilters=filters // 4, k_size=self.kernelsize, name='q_compression'),
            self.cksn(infilters=filters // 4, outfilters=pow(self.kernelsize * upscale, 3),
                      k_size=self.kernelsize, name='q_ks'),
        )
        self.k_layer = torch.nn.Sequential(
            self.cksn(infilters=filters, outfilters=filters // 4, k_size=self.kernelsize, name='k_compression'),
            self.cksn(infilters=filters // 4, outfilters=pow(upscale, 3), k_size=self.kernelsize, name='k_ks'),
        )
        self.v_layer = torch.nn.Sequential(
            self.cksn(infilters=filters, outfilters=filters * pow(self.kernelsize, 3), k_size=self.kernelsize, name='v_layer'),
            torch.nn.Upsample(scale_factor=2),
        )
        self.upscale = upscale

        self.neighbor = neighbor


    def cksn(input, infilters, outfilters, k_size, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D',
                          module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters, kernel_size=k_size,
                                                 stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, x):
        c = int(x.shape[1])
        h = int(x.shape[2])
        d = int(x.shape[3])
        w = int(x.shape[4])

        q = self.q_layer(x)
        q = PixelShuffle3d(scale=self.upscale)(q)
        q = torch.reshape(q, shape=(-1, pow(self.kernelsize, 3), h * self.upscale, d * self.upscale, w * self.upscale, 1))
        q = torch.swapaxes(q, 1, 4)

        k = self.k_layer(x)
        k = PixelShuffle3d(scale=self.upscale)(k)
        k = torch.reshape(k, shape=(-1, 1, h * self.upscale, d * self.upscale, w * self.upscale, 1))
        k = torch.swapaxes(k, 1, 4)
        qk = torch.matmul(q, k)

        qk_neighbor = torch.ones(size=(pow(self.kernelsize, 3), 1, self.neighbor, self.neighbor, self.neighbor), dtype=torch.float,
                                 device=x.device, requires_grad=False)
        qk = torch.swapaxes(qk, 1, 4)
        qk = torch.reshape(qk, shape=(-1, pow(self.kernelsize, 3), h * self.upscale, d * self.upscale, w * self.upscale))
        qk = torch.nn.functional.conv3d(qk, qk_neighbor, padding='same', groups=pow(self.kernelsize, 3)) / (self.neighbor ** 3)
        qk = torch.reshape(qk, shape=(-1, pow(self.kernelsize, 3), h * self.upscale, d * self.upscale, w * self.upscale, 1))
        qk = torch.swapaxes(qk, 1, 4)

        v = self.v_layer(x)
        v = torch.swapaxes(v, 1, 4)
        v = torch.reshape(v, shape=(-1, w * self.upscale, h * self.upscale, d * self.upscale, c, pow(self.kernelsize, 3)))
        qkv = torch.matmul(v, qk)
        qkv = torch.reshape(qkv, shape=(-1, w * self.upscale, h * self.upscale, d * self.upscale, c))
        qkv = torch.swapaxes(qkv, 1, 4)

        qkv_neighbor = torch.ones(size=(c, 1, self.neighbor, self.neighbor, self.neighbor), dtype=torch.float, device=x.device, requires_grad=False)
        qkv = torch.nn.functional.conv3d(qkv, qkv_neighbor, padding='same', groups=c) / (self.neighbor ** 3)

        return qkv

class Def_DownSampling(torch.nn.Module):
    def __init__(self, filters, neigbor):
        super(Def_DownSampling, self).__init__()
        self.convlayer = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, padding='same'),
            torch.nn.AvgPool3d(kernel_size=2, stride=2)
        )
        self.neighbor = neigbor
        self.gauslayer = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=self.neighbor, stride=1,
                                         padding='same', bias=False)
        gaussian_weight = torch.FloatTensor(self.makegaussian(self.neighbor, filters= filters))
        self.gauslayer.weight.data = gaussian_weight

    def cksn(input, infilters, outfilters, k_size, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D',
                          module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters, kernel_size=k_size,
                                                 stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def makegaussian(self, kernelsize, filters):
        center = kernelsize // 2
        x = np.zeros(shape=(kernelsize, kernelsize, kernelsize))
        y = np.zeros(shape=(kernelsize, kernelsize, kernelsize))
        z = np.zeros(shape=(kernelsize, kernelsize, kernelsize))
        for i in range(kernelsize): x[i, :, :] = i
        for j in range(kernelsize): y[:, j, :] = j
        for k in range(kernelsize): z[:, :, k] = k
        gaussiankernel = np.exp(
            -4 * np.log(2) * ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / kernelsize ** 2)
        gaussiankernel = np.expand_dims(gaussiankernel, axis=[0, 1])
        gaussiankernel = np.repeat(gaussiankernel, filters, axis=0)
        gaussiankernel = np.repeat(gaussiankernel, filters, axis=1)
        return gaussiankernel

    def forward(self, x):
        D = self.convlayer(x)
        p_filter = self.gauslayer(D)
        c = int(D.shape[1])
        h = int(D.shape[2])
        w = int(D.shape[3])
        d = int(D.shape[4])
        P = torch.zeros(size=(3, h, w, d))
        for i in range(h): P[0, i, :, :] = i / (h - 1)
        for j in range(w): P[1, :, j, :] = j / (w - 1)
        for k in range(d): P[2, :, :, k] = k / (d - 1)
        P = torch.tensor(P.expand(int(x.shape[0]), c, 3, h, w, d), requires_grad=False, device=x.device)
        D = torch.reshape(D, shape= (-1, c, 1, h, w, d))
        D_mul = torch.mul(P, D)
        D_mul = torch.swapaxes(D_mul, axis0=1, axis1=2)
        D_mul = torch.reshape(D_mul, shape=(-1, c, h, w, d))

        all_filter = self.gauslayer(D_mul)
        all_filter = torch.reshape(all_filter, shape=(-1, 3, c, h, w, d))

        x_filter = all_filter[:, 0, :, :, :, :]
        y_filter = all_filter[:, 1, :, :, :, :]
        z_filter = all_filter[:, 2, :, :, :, :]
        x_filter = (x_filter / (p_filter + 1e-6)) * 2 - 1
        y_filter = (y_filter / (p_filter + 1e-6)) * 2 - 1
        z_filter = (z_filter / (p_filter + 1e-6)) * 2 - 1

        x_grids = torch.clamp(torch.unsqueeze(x_filter, dim=1), min=-1, max=1)
        y_grids = torch.clamp(torch.unsqueeze(y_filter, dim=1), min=-1, max=1)
        z_grids = torch.clamp(torch.unsqueeze(z_filter, dim=1), min=-1, max=1)
        grids = torch.cat((x_grids, y_grids, z_grids), dim=1)
        grids = torch.swapaxes(grids, axis0=1, axis1=2)
        grids = torch.swapaxes(grids, axis0=2, axis1=3)
        grids = torch.swapaxes(grids, axis0=3, axis1=4)
        grids = torch.swapaxes(grids, axis0=4, axis1=5)

        y = torch.nn.functional.grid_sample(input = torch.unsqueeze(x[:, 0, :, :, :], dim=1), grid= grids[:, 0, :, :, :, :])
        for _ in range(1, c):
            y = torch.cat((y, torch.nn.functional.grid_sample(input = torch.unsqueeze(x[:, _, :, :, :], dim=1), grid= grids[:, _, :, :, :, :])), dim=1)
        return y, torch.reshape(D,shape=(-1, c, h, w, d)), torch.mean(grids, dim=1)

class Transfer(torch.nn.Module):
    def __init__(self, inputshape):
        super(Transfer, self).__init__()
        c = inputshape[0]
        h = inputshape[1]
        w = inputshape[2]
        d = inputshape[3]
        self.layer_1 = torch.nn.Sequential(
            self.cksn(infilters=c, outfilters=c // 2, k_size=3, k_stride=2, name='layer_1_1'),
            self.cksn(infilters=c // 2, outfilters=c // 4, k_size=3, k_stride=2, name='layer_1_2'),
        )
        self.layer_mean = torch.nn.Conv3d(in_channels=c // 4, out_channels=c, kernel_size=(h // 4, w // 4, d // 4), stride=1, padding='valid')
        self.layer_std = torch.nn.Conv3d(in_channels=c // 4, out_channels=c, kernel_size=(h // 4, w // 4, d // 4), stride=1, padding='valid')

        self.skiplayer1_1 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')
        self.skiplayer1_2 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')

        self.skiplayer2_1 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')
        self.skiplayer2_2 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')

        self.skiplayer3_1 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')
        self.skiplayer3_2 = torch.nn.Conv3d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding='same')

    def cksn(input, infilters, outfilters, k_size, k_stride, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D',
                          module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters, kernel_size=k_size,
                                                 stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        if k_stride > 1: module.add_module(name=name + 'pooling',
                                           module=torch.nn.AvgPool3d(kernel_size=k_stride, stride=k_stride))
        return module

    def AdaIn(self, feature, style_mean, style_std):
        mean = torch.mean(feature, dim=[2, 3, 4], keepdim=True)
        std = torch.std(feature, dim=[2, 3, 4], keepdim=True)
        return style_std * (feature - mean) / std + style_mean

    def forward(self, x):
        y = self.layer_1(x)
        style_mean = self.layer_mean(y)
        style_std = self.layer_std(y)

        rec_feature = self.skiplayer1_1(x)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        rec_feature = torch.nn.ReLU()(rec_feature)
        rec_feature = self.skiplayer1_2(rec_feature)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        add_feature = torch.add(x, rec_feature)

        rec_feature = self.skiplayer2_1(add_feature)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        rec_feature = torch.nn.ReLU()(rec_feature)
        rec_feature = self.skiplayer2_2(rec_feature)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        add_feature = torch.add(add_feature, rec_feature)

        rec_feature = self.skiplayer3_1(add_feature)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        rec_feature = torch.nn.ReLU()(rec_feature)
        rec_feature = self.skiplayer3_2(rec_feature)
        rec_feature = self.AdaIn(rec_feature, style_mean, style_std)
        add_feature = torch.add(add_feature, rec_feature)
        return add_feature, style_mean, style_std

class Variant_Upsampling(torch.nn.Module):
    def __init__(self, upsampling_operation):

        #如果upsampling_operation不在['nearest', 'trilinear', 'deconv', 'pixelshuffle', 'qkv']中，抛出异常
        if upsampling_operation not in ['nearest', 'trilinear', 'deconv', 'pixelshuffle', 'qkv']:
            raise ValueError('upsampling_operation should be one of [nearest, trilinear, deconv, pixelshuffle, qkv]')
        
        super(Variant_Upsampling, self).__init__()
        self.sh_enc_1 = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters=4 * Generator_filters, k_size=3, name='shenc_1'),
            self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_2'),
            Def_DownSampling(filters= 8 * Generator_filters, neigbor=5)
        )
        self.sh_enc_2 = torch.nn.Sequential(
            self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=3, name='shenc_3'),
            Def_DownSampling(filters= 16 * Generator_filters, neigbor=5)
        )

        self.sh_rsb1 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb2 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb3 = skiplayer(infilters=16 * Generator_filters)
        self.tran_1 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_2 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_3 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        
        #如果upsampling==PS, 使用Pixelshuffle3D
        if upsampling_operation == 'pixelshuffle':
            self.mr_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters * 8, k_size=3, name='mrdecoder_1'),
                PixelShuffle3d(scale=2),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters * 8, k_size=3, name='mrdecoder_2'),
                PixelShuffle3d(scale=2),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
            self.pet_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters * 8, k_size=3, name='petdecoder_1'),
                PixelShuffle3d(scale=2),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters * 8, k_size=3, name='petdecoder_2'),
                PixelShuffle3d(scale=2),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
        #如果upsampling==nearest, 使用torch.nn.Upsample
        elif upsampling_operation == 'nearest':
            self.mr_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='mrdecoder_1'),
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='mrdecoder_2'),
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
            self.pet_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
        #如果upsampling==trilinear, 使用torch.nn.Upsample
        elif upsampling_operation == 'trilinear':
            self.mr_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='mrdecoder_1'),
                torch.nn.Upsample(scale_factor=2, mode='trilinear'),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='mrdecoder_2'),
                torch.nn.Upsample(scale_factor=2, mode='trilinear'),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
            self.pet_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
                torch.nn.Upsample(scale_factor=2, mode='trilinear'),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
                torch.nn.Upsample(scale_factor=2, mode='trilinear'),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
        #如果upsampling==deconv, 使用torch.nn.ConvTranspose3d，注意第一次的目标size为(88, 72, 72),第二次为(176, 144, 144)
        elif upsampling_operation == 'deconv':
            self.mr_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                torch.nn.ConvTranspose3d(in_channels=16 * Generator_filters, out_channels=8 * Generator_filters, kernel_size=3, stride=2, padding=1),
                torch.nn.InstanceNorm3d(num_features=8 * Generator_filters),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(in_channels=8 * Generator_filters, out_channels=4 * Generator_filters, kernel_size=3, stride=2, padding=0, output_padding=1),
                torch.nn.InstanceNorm3d(num_features=4 * Generator_filters),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
            self.pet_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                torch.nn.ConvTranspose3d(in_channels=16 * Generator_filters, out_channels=8 * Generator_filters, kernel_size=3, stride=2, padding=1),
                torch.nn.InstanceNorm3d(num_features=8 * Generator_filters),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(in_channels=8 * Generator_filters, out_channels=4 * Generator_filters, kernel_size=3, stride=2, padding=0, output_padding=1),
                torch.nn.InstanceNorm3d(num_features=4 * Generator_filters),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
        elif upsampling_operation=='qkv':
            self.mr_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='mrdecoder_1'),
                QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=5),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='mrdecoder_2'),
                QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=5),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )
            self.pet_dec = torch.nn.Sequential(
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                skiplayer(infilters=16 * Generator_filters),
                self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
                QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=5),
                self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
                QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=5),
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
            )

            

    def cksn(input, infilters, outfilters, k_size, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D', module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters,
                                                                       kernel_size=k_size, stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, mrimg, petimg):
        mr_feature, mr_d1, mr_c1 = self.sh_enc_1(mrimg)
        mr_feature, mr_d2, mr_c2 = self.sh_enc_2(mr_feature)
        pet_feature, pet_d1, pet_c1 = self.sh_enc_1(petimg)
        pet_feature, pet_d2, pet_c2 = self.sh_enc_2(pet_feature)

        """
        搭建BXGAN主干运行流程
        shared_encoder->skiplayers->decoder
        """
        feature_mr_1 = self.sh_rsb1(mr_feature)
        feature_mr_2 = self.sh_rsb2(feature_mr_1)
        feature_mr_3 = self.sh_rsb3(feature_mr_2)
        feature_pet_1 = self.sh_rsb1(pet_feature)
        feature_pet_2 = self.sh_rsb2(feature_pet_1)
        feature_pet_3 = self.sh_rsb3(feature_pet_2)
        Ip1 = self.pet_dec(feature_pet_3)
        Im = self.mr_dec(feature_mr_3)

        """
        BXGANのtransfer流程
        transer->skiplayer->avg->decoder
        kl_loss对特征进行分布约束
        """
        rmr_to_pet1, p_mean1, p_std1 = self.tran_1(feature_mr_1)
        kl_1 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet1), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_1)), size_average=True)
        mr_to_pet1 = self.sh_rsb2(rmr_to_pet1)
        mr_to_pet1 = self.sh_rsb3(mr_to_pet1)
        rmr_to_pet2, p_mean2, p_std2 = self.tran_2(feature_mr_2)
        kl_2 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet2), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_2)), size_average=True)
        mr_to_pet2 = self.sh_rsb3(rmr_to_pet2)
        rmr_to_pet3, p_mean3, p_std3 = self.tran_3(feature_mr_3)
        kl_3 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet3), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_3)), size_average=True)
        mr_to_pet = (mr_to_pet1 + mr_to_pet2 + rmr_to_pet3) / 3.0
        Ip2 = self.pet_dec(mr_to_pet)

        """
        BXGANのloss
        1. 图像的mae_loss, 在主文件实现
        2. 特征的kl_loss, 在上面实现
        3. 均值标准差的reg_loss, 在下面实现
        """
        kl_loss = kl_1 + kl_2 + kl_3
        r_mean1, r_std1 = torch.mean(feature_pet_1, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_1,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean2, r_std2 = torch.mean(feature_pet_2, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_2,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean3, r_std3 = torch.mean(feature_pet_3, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_3,
                                                                                            dim=[2, 3, 4], keepdim=True)
        loss_mean1 = torch.nn.MSELoss(size_average=True)(r_mean1, p_mean1)
        loss_mean2 = torch.nn.MSELoss(size_average=True)(r_mean2, p_mean2)
        loss_mean3 = torch.nn.MSELoss(size_average=True)(r_mean3, p_mean3)
        loss_std1 = torch.nn.MSELoss(size_average=True)(r_std1, p_std1)
        loss_std2 = torch.nn.MSELoss(size_average=True)(r_std2, p_std2)
        loss_std3 = torch.nn.MSELoss(size_average=True)(r_std3, p_std3)
        reg_loss = loss_mean1 + loss_mean2 + loss_mean3 + loss_std1 + loss_std2 + loss_std3

        return Im, Ip1, Ip2, reg_loss, kl_loss, pet_d1, pet_d2, mr_d1, mr_d2, mr_c1

class Variant_pooling(torch.nn.Module):
    def __init__(self, pooling_operation):

        #如果pooling_operation不在['maxpooling', 'avgpooling','convstride']中，报错
        if pooling_operation not in ['maxpooling', 'avgpooling','convstride']:
            raise ValueError('pooling_operation must be max, avg or convstride')

        super(Variant_pooling, self).__init__()
        self.sh_enc = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters=4 * Generator_filters, k_size=3, name='shenc_1'),
            # self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_2')
        )

        if pooling_operation == 'maxpooling':
            self.sh_enc.add_module('conv1', self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_3'))
            self.sh_enc.add_module('maxpooling-1', torch.nn.MaxPool3d(kernel_size=2, stride=2))
            self.sh_enc.add_module('conv2', self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=3, name='shenc_4'))
            self.sh_enc.add_module('maxpooling-2', torch.nn.MaxPool3d(kernel_size=2, stride=2))
        elif pooling_operation == 'avgpooling':
            self.sh_enc.add_module('conv1', self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_3'))
            self.sh_enc.add_module('avgpooling-1', torch.nn.AvgPool3d(kernel_size=2, stride=2))
            self.sh_enc.add_module('conv2', self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=3, name='shenc_4'))
            self.sh_enc.add_module('avgpooling-2', torch.nn.AvgPool3d(kernel_size=2, stride=2))
        elif pooling_operation == 'convstride':
            self.sh_enc.add_module('conv1', module= torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=8 * Generator_filters, kernel_size=3, stride=2, padding=1),
                torch.nn.InstanceNorm3d(16 * Generator_filters),
                torch.nn.ReLU()
            ))
            self.sh_enc.add_module('conv2', module= torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=8 * Generator_filters, out_channels=16 * Generator_filters, kernel_size=3, stride=2, padding=1),
                torch.nn.InstanceNorm3d(16 * Generator_filters),
                torch.nn.ReLU()
            ))

          


        self.sh_rsb1 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb2 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb3 = skiplayer(infilters=16 * Generator_filters)
        self.tran_1 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_2 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_3 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        
        self.mr_dec = torch.nn.Sequential(
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='mrdecoder_1'),
            QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=5),
            self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='mrdecoder_2'),
            QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=5),
            torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
        )
        self.pet_dec = torch.nn.Sequential(
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
            QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=5),
            self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
            QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=5),
            torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
        )

            

    def cksn(input, infilters, outfilters, k_size, padding_mode = 'same', name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D', module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters,
                                                                       kernel_size=k_size, stride=1, padding=padding_mode))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, mrimg, petimg):
        mr_feature = self.sh_enc(mrimg)
        pet_feature = self.sh_enc(petimg)

        """
        搭建BXGAN主干运行流程
        shared_encoder->skiplayers->decoder
        """
        feature_mr_1 = self.sh_rsb1(mr_feature)
        feature_mr_2 = self.sh_rsb2(feature_mr_1)
        feature_mr_3 = self.sh_rsb3(feature_mr_2)
        feature_pet_1 = self.sh_rsb1(pet_feature)
        feature_pet_2 = self.sh_rsb2(feature_pet_1)
        feature_pet_3 = self.sh_rsb3(feature_pet_2)
        Ip1 = self.pet_dec(feature_pet_3)
        Im = self.mr_dec(feature_mr_3)

        """
        BXGANのtransfer流程
        transer->skiplayer->avg->decoder
        kl_loss对特征进行分布约束
        """
        rmr_to_pet1, p_mean1, p_std1 = self.tran_1(feature_mr_1)
        kl_1 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet1), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_1)), size_average=True)
        mr_to_pet1 = self.sh_rsb2(rmr_to_pet1)
        mr_to_pet1 = self.sh_rsb3(mr_to_pet1)
        rmr_to_pet2, p_mean2, p_std2 = self.tran_2(feature_mr_2)
        kl_2 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet2), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_2)), size_average=True)
        mr_to_pet2 = self.sh_rsb3(rmr_to_pet2)
        rmr_to_pet3, p_mean3, p_std3 = self.tran_3(feature_mr_3)
        kl_3 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet3), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_3)), size_average=True)
        mr_to_pet = (mr_to_pet1 + mr_to_pet2 + rmr_to_pet3) / 3.0
        Ip2 = self.pet_dec(mr_to_pet)

        """
        BXGANのloss
        1. 图像的mae_loss, 在主文件实现
        2. 特征的kl_loss, 在上面实现
        3. 均值标准差的reg_loss, 在下面实现
        """
        kl_loss = kl_1 + kl_2 + kl_3
        r_mean1, r_std1 = torch.mean(feature_pet_1, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_1,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean2, r_std2 = torch.mean(feature_pet_2, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_2,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean3, r_std3 = torch.mean(feature_pet_3, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_3,
                                                                                            dim=[2, 3, 4], keepdim=True)
        loss_mean1 = torch.nn.MSELoss(size_average=True)(r_mean1, p_mean1)
        loss_mean2 = torch.nn.MSELoss(size_average=True)(r_mean2, p_mean2)
        loss_mean3 = torch.nn.MSELoss(size_average=True)(r_mean3, p_mean3)
        loss_std1 = torch.nn.MSELoss(size_average=True)(r_std1, p_std1)
        loss_std2 = torch.nn.MSELoss(size_average=True)(r_std2, p_std2)
        loss_std3 = torch.nn.MSELoss(size_average=True)(r_std3, p_std3)
        reg_loss = loss_mean1 + loss_mean2 + loss_mean3 + loss_std1 + loss_std2 + loss_std3

        return Im, Ip1, Ip2, reg_loss, kl_loss

class Variant_Parameter(torch.nn.Module):
    def __init__(self, sigma_U = 5, sigma_D = 5):
        super(Variant_Parameter, self).__init__()
        self.sh_enc_1 = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters=4 * Generator_filters, k_size=3, name='shenc_1'),
            self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_2'),
            Def_DownSampling(filters= 8 * Generator_filters, neigbor=sigma_D)
        )
        self.sh_enc_2 = torch.nn.Sequential(
            self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=3, name='shenc_3'),
            Def_DownSampling(filters= 16 * Generator_filters, neigbor=sigma_D)
        )

        self.sh_rsb1 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb2 = skiplayer(infilters=16 * Generator_filters)
        self.sh_rsb3 = skiplayer(infilters=16 * Generator_filters)
        self.tran_1 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_2 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))
        self.tran_3 = Transfer(inputshape=(16 * Generator_filters, 44, 36, 36))

        self.mr_dec = torch.nn.Sequential(
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='mrdecoder_1'),
            QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=sigma_U),
            self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='mrdecoder_2'),
            QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=sigma_U),
            torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
        )
        self.pet_dec = torch.nn.Sequential(
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
            QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=sigma_U),
            self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
            QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=sigma_U),
            torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
        )


    def cksn(input, infilters, outfilters, k_size, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D', module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters,
                                                                       kernel_size=k_size, stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, mrimg, petimg):
        mr_feature, mr_d1, mr_c1 = self.sh_enc_1(mrimg)
        mr_feature, mr_d2, mr_c2 = self.sh_enc_2(mr_feature)
        pet_feature, pet_d1, pet_c1 = self.sh_enc_1(petimg)
        pet_feature, pet_d2, pet_c2 = self.sh_enc_2(pet_feature)

        """
        搭建BXGAN主干运行流程
        shared_encoder->skiplayers->decoder
        """
        feature_mr_1 = self.sh_rsb1(mr_feature)
        feature_mr_2 = self.sh_rsb2(feature_mr_1)
        feature_mr_3 = self.sh_rsb3(feature_mr_2)
        feature_pet_1 = self.sh_rsb1(pet_feature)
        feature_pet_2 = self.sh_rsb2(feature_pet_1)
        feature_pet_3 = self.sh_rsb3(feature_pet_2)
        Ip1 = self.pet_dec(feature_pet_3)
        Im = self.mr_dec(feature_mr_3)

        """
        BXGANのtransfer流程
        transer->skiplayer->avg->decoder
        kl_loss对特征进行分布约束
        """
        rmr_to_pet1, p_mean1, p_std1 = self.tran_1(feature_mr_1)
        kl_1 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet1), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_1)), size_average=True)
        mr_to_pet1 = self.sh_rsb2(rmr_to_pet1)
        mr_to_pet1 = self.sh_rsb3(mr_to_pet1)
        rmr_to_pet2, p_mean2, p_std2 = self.tran_2(feature_mr_2)
        kl_2 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet2), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_2)), size_average=True)
        mr_to_pet2 = self.sh_rsb3(rmr_to_pet2)
        rmr_to_pet3, p_mean3, p_std3 = self.tran_3(feature_mr_3)
        kl_3 = kl_div(torch.log_softmax(torch.nn.Flatten()(rmr_to_pet3), dim=1),
                      torch.nn.Softmax()(torch.nn.Flatten()(feature_pet_3)), size_average=True)
        mr_to_pet = (mr_to_pet1 + mr_to_pet2 + rmr_to_pet3) / 3.0
        Ip2 = self.pet_dec(mr_to_pet)

        """
        BXGANのloss
        1. 图像的mae_loss, 在主文件实现
        2. 特征的kl_loss, 在上面实现
        3. 均值标准差的reg_loss, 在下面实现
        """
        kl_loss = kl_1 + kl_2 + kl_3
        r_mean1, r_std1 = torch.mean(feature_pet_1, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_1,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean2, r_std2 = torch.mean(feature_pet_2, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_2,
                                                                                            dim=[2, 3, 4], keepdim=True)
        r_mean3, r_std3 = torch.mean(feature_pet_3, dim=[2, 3, 4], keepdim=True), torch.std(feature_pet_3,
                                                                                            dim=[2, 3, 4], keepdim=True)
        loss_mean1 = torch.nn.MSELoss(size_average=True)(r_mean1, p_mean1)
        loss_mean2 = torch.nn.MSELoss(size_average=True)(r_mean2, p_mean2)
        loss_mean3 = torch.nn.MSELoss(size_average=True)(r_mean3, p_mean3)
        loss_std1 = torch.nn.MSELoss(size_average=True)(r_std1, p_std1)
        loss_std2 = torch.nn.MSELoss(size_average=True)(r_std2, p_std2)
        loss_std3 = torch.nn.MSELoss(size_average=True)(r_std3, p_std3)
        reg_loss = loss_mean1 + loss_mean2 + loss_mean3 + loss_std1 + loss_std2 + loss_std3

        return Im, Ip1, Ip2, reg_loss, kl_loss, pet_d1, pet_d2, mr_d1, mr_d2, mr_c1


class Variant_Direct(torch.nn.Module):
    def __init__(self):
        super(Variant_Direct, self).__init__()
        self.enc_1 = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters=4 * Generator_filters, k_size=3, name='shenc_1'),
            self.cksn(infilters=4 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='shenc_2'),
            Def_DownSampling(filters= 8 * Generator_filters, neigbor=5)
        )
        self.enc_2 = torch.nn.Sequential(
            self.cksn(infilters=8 * Generator_filters, outfilters=16 * Generator_filters, k_size=3, name='shenc_3'),
            Def_DownSampling(filters= 16 * Generator_filters, neigbor=5)
        )

        self.pet_dec = torch.nn.Sequential(
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            skiplayer(infilters=16 * Generator_filters),
            self.cksn(infilters=16 * Generator_filters, outfilters=8 * Generator_filters, k_size=3, name='petdecoder_1'),
            QKV_Upsampling(filters= 8 * Generator_filters, upscale=2, neighbor=5),
            self.cksn(infilters=8 * Generator_filters, outfilters=4 * Generator_filters, k_size=3, name='petdecoder_2'),
            QKV_Upsampling(filters=4 * Generator_filters, upscale=2, neighbor=5),
            torch.nn.Conv3d(in_channels=4 * Generator_filters, out_channels=1, kernel_size=7, padding='same')
        )

    def cksn(input, infilters, outfilters, k_size, name=None):
        module = torch.nn.Sequential()
        module.add_module(name=name + 'conv3D', module=torch.nn.Conv3d(in_channels=infilters, out_channels=outfilters,
                                                                       kernel_size=k_size, stride=1, padding='same'))
        module.add_module(name=name + 'IN', module=torch.nn.InstanceNorm3d(num_features=outfilters))
        module.add_module(name=name + 'activation', module=torch.nn.ReLU())
        return module

    def forward(self, mrimg):
        mr_feature, mr_d1, mr_c1 = self.enc_1(mrimg)
        mr_feature, mr_d2, mr_c2 = self.enc_2(mr_feature)

        Ip1 = self.pet_dec(mr_feature)

        return Ip1, mr_d1, mr_d2, mr_c1


filters = 2
class MFCN(torch.nn.Module):
    def cksn(self, infilters, outfilters, k_size, k_stride, padding_mode = 'same', name = None):
        module = torch.nn.Sequential()
        module.add_module(name = name + 'conv3D',
                          module= torch.nn.Conv3d(in_channels= infilters, out_channels= outfilters, kernel_size= k_size,
                                                  stride= k_stride, padding=padding_mode))
        module.add_module(name = name + 'BN', module= torch.nn.BatchNorm3d(num_features= outfilters))
        module.add_module(name = name + 'activation', module = torch.nn.ReLU())
        return module

    def __init__(self, number_class):
        super(MFCN, self).__init__()
        self.m1 = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters= 16 * filters, k_size=3, k_stride=1, name= 'm1_1'),
            self.cksn(infilters=16 * filters, outfilters= 16 * filters, k_size=3, k_stride=1, name='m1_2'),
            torch.nn.MaxPool3d(2)
        )
        self.p1 = torch.nn.Sequential(
            self.cksn(infilters=1, outfilters= 16 * filters, k_size=3, k_stride=1, name= 'p1_1'),
            self.cksn(infilters=16 * filters, outfilters= 16 * filters, k_size=3, k_stride=1, name='p1_2'),
            torch.nn.MaxPool3d(2)
        )
        # self.t1 = MKM(inputsize=(1, 16*filters, 88, 72, 72))

        self.m2 = torch.nn.Sequential(
            DenseBlock(channels= 16 * filters),
            self.cksn(infilters= 64 * filters, outfilters= 32 * filters, k_size=1, k_stride=1, name='m2_1'),
            torch.nn.AvgPool3d(2)
        )
        self.p2 = torch.nn.Sequential(
            DenseBlock(channels= 16 * filters),
            self.cksn(infilters= 64 * filters, outfilters= 32 * filters, k_size=1, k_stride=1, name='p2_1'),
            torch.nn.AvgPool3d(2)
        )
        # self.t2 = MKM(inputsize=(1, 32*filters, 44, 36, 36))

        self.m3 = torch.nn.Sequential(
            DenseBlock(channels= 32 * filters),
            self.cksn(infilters= 128 * filters, outfilters= 64 * filters, k_size=1, k_stride=1, name='m3_1'),
            torch.nn.AvgPool3d(2)
        )
        self.p3 = torch.nn.Sequential(
            DenseBlock(channels= 32 * filters),
            self.cksn(infilters= 128 * filters, outfilters= 64 * filters, k_size=1, k_stride=1, name='p3_1'),
            torch.nn.AvgPool3d(2)
        )
        self.t3 = MKM(inputsize=(1, 64*filters, 22, 18, 18))

        # self.m4 = DenseBlock(channels= 64 * filters)
        # self.p4 = DenseBlock(channels= 64 * filters)

        self.c_m = torch.nn.Sequential(
            self.cksn(infilters= 64 * filters, outfilters= 8 * filters, k_size=3, k_stride=1, name= 'c1'),
            self.cksn(infilters= 8 * filters, outfilters= 1, k_size=3, k_stride=1, name='c2'),
            torch.nn.MaxPool3d(2),
            torch.nn.Flatten(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features= 891, out_features=number_class),
            torch.nn.Softmax()
        )
        self.c_p = torch.nn.Sequential(
            self.cksn(infilters= 64 * filters, outfilters= 8 * filters, k_size=3, k_stride=1, name= 'c1'),
            self.cksn(infilters= 8 * filters, outfilters= 1, k_size=3, k_stride=1, name='c2'),
            torch.nn.MaxPool3d(2),
            torch.nn.Flatten(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features= 891, out_features=number_class),
            torch.nn.Softmax()
        )

    def forward(self, mr, pet):
        mf = self.m1(mr)
        pf = self.p1(pet)

        mf = self.m2(mf)
        pf = self.p2(pf)

        mf = self.m3(mf)
        pf = self.p3(pf)
        f = self.t3(mf, pf)
        f = torch.reshape(f, shape=(64 * filters, 1, 3, 3, 3))
        mf = F.conv3d(mf, f, stride=1, padding='same', groups=64 * filters)
        pf = F.conv3d(pf, f, stride=1, padding='same', groups=64 * filters)

        mm = self.c_m(mf)
        mp = self.c_p(mf)
        pm = self.c_m(pf)
        pp = self.c_p(pf)

        return mm, mp, pm, pp

class MKM(torch.nn.Module):
    def cksn(self, infilters, outfilters, k_size, k_stride, padding_mode = 'same', name = None):
        module = torch.nn.Sequential()
        module.add_module(name = name + 'conv3D',
                          module= torch.nn.Conv3d(in_channels= infilters, out_channels= outfilters, kernel_size= k_size,
                                                  stride= k_stride, padding=padding_mode))
        module.add_module(name = name + 'BN', module= torch.nn.BatchNorm3d(num_features= outfilters))
        module.add_module(name = name + 'activation', module = torch.nn.ReLU())
        return module

    def __init__(self, inputsize):
        super(MKM, self).__init__()
        batch, channel, height, width, depth = inputsize
        self.mlayer = self.cksn(infilters= channel, outfilters= channel, k_size=1, k_stride=1, padding_mode='same', name='mlayer')
        self.player = self.cksn(infilters= channel, outfilters= channel, k_size=1, k_stride=1, padding_mode='same', name='player')
        self.flayer = self.cksn(infilters= 2 * channel, outfilters= channel, k_size=(height//2, width//2, depth//2),
                                k_stride=(height//4, width//4, depth//4), padding_mode='valid', name='flayer')

    def forward(self, mr, pet):
        mf = self.mlayer(mr)
        pf = self.player(pet)
        f = torch.cat((mf, pf), dim=1)
        kernel = self.flayer(f)
        return kernel

class DenseBlock(torch.nn.Module):

    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.l1 = self.cksn(infilters= channels, outfilters= channels, k_size=3, k_stride=1, name='l1')
        self.l2 = self.cksn(infilters= 2 * channels, outfilters= channels, k_size=3, k_stride=1, name='l2')
        self.l3 = self.cksn(infilters= 3 * channels, outfilters= channels, k_size=3, k_stride=1, name='l3')

    def cksn(self, infilters, outfilters, k_size, k_stride, name = None):
        module = torch.nn.Sequential()
        module.add_module(name = name + 'conv3D',
                          module= torch.nn.Conv3d(in_channels= infilters, out_channels= outfilters, kernel_size= k_size,
                                                  stride= k_stride, padding='same'))
        module.add_module(name = name + 'BN', module= torch.nn.BatchNorm3d(num_features= outfilters))
        module.add_module(name = name + 'activation', module = torch.nn.ReLU())
        return module

    def forward(self, f):
        f1 = self.l1(f)
        f = torch.cat((f, f1), dim=1)
        f2 = self.l2(f)
        f = torch.cat((f, f2), dim=1)
        f3 = self.l3(f)
        f = torch.cat((f, f3), dim=1)
        return f
