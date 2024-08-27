import os
#import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse
import numpy
import torch
import torch.utils.data as data

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def parse_opt():
#Set train options
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/pyramidpix2pix', help='results saved path')
    opt = parser.parse_args()
    return opt


class testDataloader(data.Dataset):
    def __init__(self, result_path):
        self.result_path = result_path
        self.result_paths = []
        for i in tqdm(os.listdir(os.path.join(result_path,'test_latest/images'))):
            if 'fake_B' in i:
                self.result_paths.append(i)

    def __len__(self):
        return len(self.result_paths)
    
    def __getitem__(self, index):
        i = self.result_paths[index]
        fake = numpy.array(Image.open(os.path.join(self.result_path,'test_latest/images',i)))
        real = numpy.array(Image.open(os.path.join(self.result_path,'test_latest/images',i.replace('fake_B','real_B'))))

        fake_norm = torch.tensor((fake-fake.min())/(fake.max()-fake.min())).permute(2,0,1).float()
        real_norm = torch.tensor((real- real.min())/(real.max()-real.min())).permute(2,0,1).float()

        return fake_norm, real_norm

def lpips_is_fid(result_path):

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    inception = InceptionScore(normalize=True)

    dataset = testDataloader(result_path)
    #Create a dataloader with 64 batch size
    batch_size = 64
    # No transform
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ssims = []
    lps = []

    for fake_norm, real_norm in tqdm(dataloader):
        ms = ms_ssim(fake_norm,real_norm).item()
        lp = lpips(fake_norm,real_norm).item()
        
        fid.update(real_norm, real=True)
        fid.update(fake_norm, real=False)
        inception.update(real_norm)

        ssims.append(ms)
        lps.append(lp)

    # write to file
    with open(os.path.join(result_path,'results.txt'), "a+") as f:
        f.write("The average ms_ssim is " + str(numpy.average(numpy.array(ssims)))+'\n')
        f.write("The average lpips is " + str(numpy.average(numpy.array(lps)))+'\n')
        f.write("The average IS cond is " + str(inception.compute()[0].item())+'\n')
        f.write("The average IS marg is " + str(inception.compute()[1].item())+'\n')
        f.write("The average FID is " + str(fid.compute().item())+'\n')

def psnr_and_ssim(result_path):
    psnr = []
    ssim = []
    for i in tqdm(os.listdir(os.path.join(result_path,'test_latest/images'))):
        if 'fake_B' in i:
            
                fake = numpy.array(Image.open(os.path.join(result_path,'test_latest/images',i)))
                real = numpy.array(Image.open(os.path.join(result_path,'test_latest/images',i.replace('fake_B','real_B'))))
                PSNR = peak_signal_noise_ratio(fake, real)

                psnr.append(PSNR)
                SSIM = structural_similarity(fake, real, channel_axis=2)
                ssim.append(SSIM)
        else:
            continue
    average_psnr=sum(psnr)/len(psnr)
    average_ssim=sum(ssim)/len(ssim)
    with open(os.path.join(result_path,'results.txt'), "a+") as f:
        f.write("The average psnr is " + str(average_psnr)+'\n')
        f.write("The average ssim is " + str(average_ssim)+'\n')

if __name__ == '__main__':
    opt = parse_opt()
    #print("Testing PSNR and SSIM...")
    #psnr_and_ssim(opt.result_path)
    print("Testing FID, IS, and LPIPS, MSSSIM")
    lpips_is_fid(opt.result_path)