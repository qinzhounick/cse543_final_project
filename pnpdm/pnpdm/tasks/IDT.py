import h5py
import torch
import numpy as np
import torch.nn as nn
# import mat73
import hdf5storage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
from . import register_operator
########################################################Functions#################################################
FFT = lambda x: torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x), dim=(-2, -1)))
iFFT = lambda x: torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x), dim=(-2, -1)))
rFFT = lambda x: torch.rfft(x, signal_ndim=2, onesided=False)


def tensor_normlize(n_ipt:torch.tensor):
    if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
        dim_x, dim_y = n_ipt.shape[2:4]
        n_ipt_max = torch.max(n_ipt.max(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
        n_ipt_min = torch.min(n_ipt.min(dim=2)[0],dim=2)[0].repeat(dim_x,dim_y,1,1).permute(2,3,0,1)
        # n_ipt_norm = (n_ipt - n_ipt_min) / (n_ipt_max - n_ipt_min)
        # Normalize to [-1, 1]
        n_ipt_norm = 2.0 * ((n_ipt - n_ipt_min) / (n_ipt_max - n_ipt_min + 1e-8 )) - 1.0
    return n_ipt_norm, n_ipt_max, n_ipt_min

def save_gray_img(img_matr,path='/export/project/w.weining/pnpdm/pnpdm/results/process/grayscale_image_corrupted.png'):
    # Convert the PyTorch tensor to a NumPy array
    print(img_matr.shape)
    grayscale_img_numpy = img_matr.squeeze(0).cpu().numpy()  # Remove the batch dimension and move to CPU

    # # Save using PIL
    # img_pil = Image.fromarray((grayscale_img_numpy * 255).astype(np.uint8))  # Convert to uint8 for saving as an image
    # img_pil.save('/export/project/w.weining/pnpdm/pnpdm/results/y_n/grayscale_image.png')

    # Alternatively, you can save using Matplotlib
    
    plt.imsave(path, grayscale_img_numpy, cmap='gray')
    print(f"sucessfully save in {path}")


# def tensor_normlize(n_ipt: torch.tensor):
#     if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
#         dim_x, dim_y = n_ipt.shape[2:4]
#         n_ipt_max = torch.max(n_ipt.max(dim=2)[0], dim=2)[0].repeat(dim_x, dim_y, 1, 1).permute(2, 3, 0, 1)
#         n_ipt_min = torch.min(n_ipt.min(dim=2)[0], dim=2)[0].repeat(dim_x, dim_y, 1, 1).permute(2, 3, 0, 1)
#         n_ipt_norm = (n_ipt - n_ipt_min) / (n_ipt_max - n_ipt_min)
#     return n_ipt_norm, n_ipt_max, n_ipt_min


def tensor_denormlize(n_ipt, n_ipt_max, n_ipt_min):
    if len(n_ipt.shape) == 4 and n_ipt.shape[2] == n_ipt.shape[3]:
        dim_x, dim_y = n_ipt.shape[2:4]
        n_ipt_denorm = torch.mul(n_ipt, n_ipt_max - n_ipt_min) + n_ipt_min
    return n_ipt_denorm


def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):

    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    ######## replace ment


    return torch.stack([res_real, res_imag], -1)


def addwgn_torch(x: torch.Tensor, inputSnr):
    noiseNorm = torch.norm(x.flatten() * 10 ** (-inputSnr / 20))
    # print(noiseNorm)
    noise = torch.randn(x.shape[-2], x.shape[-1]).to(noiseNorm.device)
    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    rec_y = x + noise.to(x.device)#.cuda()

    return rec_y, noise

def complex_divided_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)  # a, b
    y_real, y_imag = torch.unbind(y, -1)  # c, d
    base_ = torch.pow(y_real, 2) + torch.pow(y_imag, 2)
    res_real = (torch.mul(x_real, y_real) + torch.mul(x_imag, y_imag)) / base_
    res_imag = (torch.mul(x_imag, y_real) - torch.mul(x_real, y_imag)) / base_
    return torch.stack([res_real, res_imag], -1)


def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)


##############################################################################################
FFT  = lambda x: torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x), dim = (-2,-1)))
iFFT = lambda x: torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x), dim = (-2,-1)))
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)


@register_operator(name='IDT_operator')
class IDTClass(nn.Module):
    def initialize(self, gt, y):
        return torch.zeros_like(gt)

    def __init__(self,opt,device=None,img_H= None):
        print("opt,device",opt,device,type(device))
        super(IDTClass, self).__init__()
        
        IDTData = hdf5storage.loadmat(opt.meas_path)
        self.Himg_temp = torch.cat((torch.tensor(IDTData['Himag'][0, 0]), torch.tensor(IDTData['Himag'][0, 1])),dim=3).permute([3, 2, 0, 1])
        # print(self.Himg_temp.shape)
        # pip gpus(IDTData['Hreal'][0, 1].shape)
        
        # print(self.Hreal_temp.shape)
        self.opt = opt
        self.opt.device = "cuda:2"
        self.Hreal_temp = torch.cat((torch.tensor(IDTData['Hreal'][0,0]), torch.tensor(IDTData['Hreal'][0,1])), dim = 3).permute([3, 2, 0, 1, 4]).to(self.opt.device)
        if img_H is not None:
            self.y = self.get_y(img_H)

    def get_init(self,ipt):
        ipt_real = torch.unbind(iFFT(ipt), -1)[0]
        ipt_real, max_, min_ = tensor_normlize(ipt_real)
        ipt = torch.stack([ipt_real, torch.zeros_like(ipt_real)], -1)
        ipt = FFT(ipt)
        return ipt

    def get_y(self,x):
        ph_ = x
        ab_ = torch.zeros_like(ph_)
        ph_ = torch.stack([ph_, torch.zeros_like(ph_)], -1)  # [1 90 90 2]
        ab_ = torch.stack([ab_, torch.zeros_like(ab_)], -1)
        intensityPred = torch.unbind(iFFT(IDTClass.fmult(FFT(ph_), FFT(ab_), self.Hreal_temp)), -1)[0]
        data = intensityPred
        if self.opt.add_noise:
            for j in range(0, self.opt.NBF_KEEP):
                # data[j, 0] = addwgn_torch(intensityPred[j, 0], opt.inputSNR)[0]

                data[j, 0] = intensityPred[j, 0] + self.opt.input_noise * (
                        torch.amax(intensityPred) - torch.amin(intensityPred)) * torch.randn_like(
                    intensityPred[j, 0]).to(intensityPred.device)

        else:
            data = intensityPred

        data = torch.stack([data, torch.zeros_like(data)], -1).to(x.device)
        y_each = FFT(data)  # torch.Size([60, 1, 90, 90, 2])
        return y_each


    def size(self):
        return self.ipt.size()



    def forward(self,x):
        opt = self.opt
        ph_ = x
        ab_ = torch.zeros_like(ph_)
        ph_ = torch.stack([ph_, torch.zeros_like(ph_)], -1)  # [1 90 90 2]
        ab_ = torch.stack([ab_, torch.zeros_like(ab_)], -1)
        intensityPred = torch.unbind(iFFT(IDTClass.fmult(FFT(ph_), FFT(ab_), self.Hreal_temp)), -1)[0]# A*x
        data = torch.zeros((opt.NBF_KEEP, 1, opt.IMG_Patch[1], opt.IMG_Patch[2]), dtype=torch.float32)
        # to replace NAN value with zero
        intensityPred = torch.nan_to_num(intensityPred, nan=0.0)
        if opt.add_noise:
            for j in tqdm(range(opt.NBF_KEEP)):
                # data[j, 0] = addwgn_torch(intensityPred[j, 0], opt.inputSNR)[0]
                # print("noise marix:",(torch.amax(intensityPred) ).to(intensityPred.device))
                data[j, 0] = intensityPred[j, 0] + self.opt.input_noise* (
                            torch.amax(intensityPred) - torch.amin(intensityPred)) * torch.randn_like(
                    intensityPred[j, 0]).to(intensityPred.device)
        else:
            data = intensityPred
   
        data = torch.stack([data, torch.zeros_like(data)], -1).to(x.device) # A*x +e
        y_each = FFT(data)
        ipt_each_real = torch.unbind(iFFT(IDTClass.ftran(y_each, self.Hreal_temp, 'Ph', opt.NBF_KEEP)), -1)[0]
        # ipt_each_real[ipt_each_real <= 0] = 0
        ipt_each_real_normalized, _, _ = tensor_normlize(ipt_each_real.unsqueeze(0))
        # print("ipt_each_real_normalized, y_each",ipt_each_real_normalized, y_each, ipt_each_real.shape,ipt_each_real_normalized.shape, y_each.shape,"||||") #ipt_each_real[1,256,256],y_each[500, 1, 256, 256, 2]
        return y_each,ipt_each_real_normalized



    def fgrad_sto(self, f_ipt, f_y=None, meas_list=None, emParams_cuda=None):
        f_ipt = torch.stack([f_ipt, torch.zeros_like(f_ipt)], -1)
        f_ph = FFT(f_ipt)  # ([1, 1, 416, 416, 2])
        f_ab = torch.zeros_like(f_ph)  # [1, 1, 416, 416, 2]


        Hreal_1 = 28
        Hreal_2 = 472

        sub1 = torch.randperm(Hreal_1)
        sub2 = torch.randperm(Hreal_2)
        meas_list_1 = sub1[0:Hreal_1]
        meas_list_2 = sub2[0:self.opt.batch_NBF - Hreal_1]
        meas_list = torch.cat([meas_list_1, (meas_list_2 + Hreal_1)])

        Hreal_temp = self.Hreal_temp[meas_list, :].to(f_ph.device)
        f_y = f_y[ meas_list, ...].to(f_ph.device)  # [1, 500, 1, 416, 416, 2])
        # print("data shape in :", f_ph.shape[0])
        # print("measurement shape in:", f_y.shape)
        # # gradPhList = []
        # for i in range(f_ph.shape[0]):
        #     gradPhList.append(self.gradPhStoc(f_ph[i], f_ab[i], meas_list, Hreal_temp, f_y[i], self.opt.batch_NBF))
        #
        # f_grad = torch.stack(gradPhList, 0)
        f_grad = self.gradPhStoc(f_ph, f_ab, meas_list, Hreal_temp, f_y, self.opt.batch_NBF)
        return f_grad  # [1, 1, 416, 416, 2]

   

   

    def prox_generator(self, f_ipt, f_y=None, emParams_cuda=None, rho=0.01, gamma=1e-2):
        f_ph = f_ipt
        
        emParams_cuda = self.Hreal_temp
        # try to normalize this f_ipt
        f_ipt = FFT(torch.stack([f_ipt, torch.zeros_like(f_ipt)], -1))
        # print("f_y",f_y[0].shape,f_y.shape,IDTClass.fdig(emParams_cuda,self.opt.NBF_KEEP).shape)
        
        numerator = gamma * IDTClass.ftran(f_y, emParams_cuda,'Ph',self.opt.NBF_KEEP) + f_ipt / (rho ** 2)
      
        self.identity = torch.stack([torch.ones_like(numerator[..., 0]), torch.zeros_like(numerator[..., 1])],
                                        -1).to(self.opt.device)
        denominator = self.identity / (rho ** 2) + gamma * IDTClass.fdig(emParams_cuda,self.opt.NBF_KEEP)
        # print("numerator,denominator",numerator.shape,denominator.shape)
        fprox_eff = complex_divided_torch(numerator, denominator)
        # print("fprox_eff",fprox_eff.shape)
        z_real = torch.unbind(iFFT(fprox_eff), -1)[0]
        # z_real[z_real<=0]=0
        # z_real, max_, min_ = tensor_normlize(z_real)
        return z_real
    @staticmethod
    def gradPh(ph, ab, emParams, y):
        z = IDTClass.fmult(ph, ab, emParams)
        g = IDTClass.ftran(z - y, emParams, 'Ph')
        return g

    @staticmethod
    def gradAb(ph, ab, emParams, y):
        z = IDTClass.fmult(ph, ab, emParams)
        g = IDTClass.ftran(z - y, emParams, 'Ab')
        return g

    @staticmethod
    def gradPhStoc(ph, ab, meas_list, emStoc, yStoc, NBFKEEP):
        zStoc = IDTClass.fmult(ph, ab, emStoc)
        g = IDTClass.ftran(zStoc - yStoc, emStoc, 'Ph',NBFKEEP)
        return g

    @staticmethod
    def fmult(ph, ab, Hreal):
        # print(ph.shape)
        # ph = ph.unsqueeze_(0)
        ph = ph.expand(size=(Hreal.shape[0],) + ph.shape[1:])
        if ph.shape != Hreal.shape:
            if len(ph.shape) > len(Hreal.shape):
                ph = ph.squeeze(1)
            else:
                ph = ph.unsqueeze_(1)
        # print(ph.shape)
        # print(Hreal.shape)
        # ab = ab.unsqueeze_(0)
        # ab = ab.expand(size=(Hreal.shape(0),) + ab.shape[1:])
        # print("Hreal, ph",Hreal.shape, ph.shape)
        z = complex_multiple_torch(Hreal, ph)  # + complex_multiple_torch(emParams['Himag'], ab)
        # print("shape of out",z.shape)

        return z

    @staticmethod
    def ftran(z, H, which, NBFKEEP):
        assert which in ['Ph', 'Ab'], "Error in which"
        if which == 'Ph': # take the complex conjugate and perform complex multiplication
            Hreal_real, Hreal_imag = torch.unbind(H, -1)
            Hreal_imag = -Hreal_imag
            # replace nan with zero(modified by Jerry)
            Hreal_real = torch.nan_to_num(Hreal_real, nan=0.0)
            Hreal_imag = torch.nan_to_num(Hreal_imag, nan=0.0)
            Hreal = torch.stack([Hreal_real, Hreal_imag], -1)
            # print("Hreal",Hreal,Hreal.shape,z,z.shape)
            x = torch.sum(complex_multiple_torch(Hreal, z), 0)
            
        else:
            Himag = H
            Himag_real, Himag_imag = torch.unbind(Himag, -1)
            Himag_imag = -Himag_imag
            Himag = torch.stack([Himag_real, Himag_imag], -1)

            x = torch.sum(complex_multiple_torch(Himag, z), 0)
        x = x / NBFKEEP
        # print("x shape:", x.shape,x)
        return x

    @staticmethod
    def fdig(emParams,NBF_KEEP):
        Hreal = emParams #modified
        Hreal = torch.nan_to_num(Hreal, nan=0.0)
        Hreal_real, Hreal_imag = torch.unbind(Hreal, -1)
        Hreal_conj = torch.stack([Hreal_real, -Hreal_imag], -1)
        x = torch.sum(complex_multiple_torch(Hreal_conj, Hreal), 0, keepdim=True)
        # print("fdig",x,x.shape)
        x = x / NBF_KEEP #modified

        return x

    @property
    def display_name(self):
        return 'IDT'


class index_choose_():
    def __init__(self, index_sets):

        if len(index_sets) != 0:

            self.set_indx = index_sets['angle_index']
            self.used_index = index_sets['used_index'] - 1
            self.angle_lst = []

            for i in range(self.set_indx.shape[0]):
                set_indx_temp = self.set_indx[i]
                angle_lst_sub = []

                for j in range(set_indx_temp.shape[0]):
                    if set_indx_temp[j] != 0:
                        angle_lst_sub.append(set_indx_temp[j] - 1)

                self.angle_lst.append(angle_lst_sub)
        else:
            pass

    def get_subset_radial(self, batch_size=3):

        sub = np.random.choice(len(self.angle_lst), batch_size, replace=False)
        # print(sub)
        sub_list = []
        for i in sub:
            sub_list = sub_list + self.angle_lst[i]
            # print(len(sub_list))
        seen = set()
        uniq = []
        for x in sub_list:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        uniq.sort()
        sub_list = uniq
        # print(sub_list)
        sub_list_all = [np.where(self.used_index == i) for i in sub_list]

        sub_list_all = np.concatenate(sub_list_all, 0).squeeze()

        return sub_list_all

    @staticmethod
    def get_subset_uniform(NBFkeep=92, batch_size=30, num_div=5):

        sub = torch.randperm(NBFkeep // num_div)[0:batch_size // num_div]
        sub, _ = torch.sort(sub)
        meas_list = torch.cat([sub + i * NBFkeep // num_div for i in range(num_div)]).tolist()
        meas_list.sort()
        return meas_list

    @staticmethod
    def get_subset_random(NBFkeep=92, batch_size=30):
        sub = torch.randperm(NBFkeep).tolist()
        meas_list = sub[0:batch_size]
        meas_list.sort()
        return meas_list
