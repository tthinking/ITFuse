from PIL import Image
import numpy as np
import os
import torch

import imageio

import torchvision.transforms as transforms

from Networks.net import MODEL as net


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=1)

model_path = ".\models\model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))
    print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)


def fusion_gray():


    path1 = './images/ir/IR_001.bmp'

    path2 = './images/vi/VIS_001.bmp'

    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')


    img1_org = img1
    img2_org = img2

    tran = transforms.ToTensor()

    img1_org = tran(img1_org)
    img2_org = tran(img2_org)


    img1_org= img1_org.cuda()
    img2_org = img2_org.cuda()


    img1_org = img1_org.unsqueeze(0).cuda()
    img2_org = img2_org.unsqueeze(0).cuda()
    model.eval()

    out = model(img1_org ,img2_org )


    out = np.squeeze(out.detach().cpu().numpy())

    out = (out * 255).astype(np.uint8)

    imageio.imwrite('./result/001.bmp', out)

if __name__ == '__main__':

    fusion_gray()
