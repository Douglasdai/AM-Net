#CUDA_VISIBLE_DEVICES=X python train.py --cuda --outpath ./outputs
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import AttU_Net
from LoadData import Dataset, loader, Dataset_val
import logging
import time
import glob
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
parser.add_argument('--outpath', default='./output_test', help='folder to output images and model checkpoints')
opt = parser.parse_args()

print(opt)


try:
    os.makedirs(opt.outpath)
except OSError:
    pass

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))+1e-6

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total

def to_float_and_cuda(input):
    input = input.type(torch.FloatTensor)
    input = input.cuda()
    return input


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
print('===> Building model')
NetS = AttU_Net()
# NetC = NetC(ngpu = opt.ngpu)


if cuda:
    NetS = NetS.cuda()
    #NetC = NetC.cuda()

# setup optimizer
lr = opt.lr
decay = opt.decay
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
test_data = glob.glob('../data_5pictures/images_5/*.jpg')
test_label = glob.glob('../data_5pictures/masks_5/*.jpg')
test_lung = glob.glob('../data_5pictures/lungs_5/*.jpg')
test_med = glob.glob('../data_5pictures/med_5/*.jpg')
#dataloader = loader(Dataset(test_data),opt.batchSize)
dataloader_val = loader(Dataset_val(test_data,test_label,test_lung,test_med), opt.batchSize)

max_iou = 0
NetS.train()
NetS.load_state_dict(torch.load('./outputs_att_info/NetS_epoch_137_5.pth'))
print('load S ok')    
NetS.eval()

# NetC.train()
# NetC.load_state_dict(torch.load('./outputs/NetC_epoch_220.pth'))
# print('load C ok')    
# NetC.eval()


#分析训练数据
for i, data in enumerate(dataloader_val, 1):
    input, gt ,lung, med,name = Variable(data[0]), Variable(data[1]),Variable(data[2]) ,Variable(data[3]),data[4]
    #input, gt,name = Variable(data[0]), Variable(data[1]),data[2] # [24, 1, 96, 96])
    name = np.array(name)
    if cuda:
        input = input.cuda()
        gt = gt.cuda()
        lung = lung.cuda()
        med = med.cuda()
        input  =to_float_and_cuda(input)
        gt =to_float_and_cuda(gt)
        med = to_float_and_cuda(med)
        lung = to_float_and_cuda(lung)

    pred,text, spicu, lobu, margin, spher, calci, sub, meter, mali = NetS(input,lung,med)
    #pred = NetS(input)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    pred = pred.type(torch.LongTensor)
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    for x in range(input.size()[0]):
        IoU = np.sum(pred_np[x][gt[x]==1]) / (float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))+1e-6)
        dice = np.sum(pred_np[x][gt[x]==1])*2 / (float(np.sum(pred_np[x]) + np.sum(gt[x]))+1e-6)
        # IoUs.append(IoU)
        # dices.append(dice)
        
        # for i in range(len(IoUs)):
        #     if IoUs[i] =='nan':
        #         IoUs[i] = 0
        #         dices[i]= 0 
        if IoU == 0 :
            continue
        else:
            vutils.save_image(data[0][x],'output_5/'+str(name[x][:-4])+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_.jpg',normalize=True)
            vutils.save_image(data[1][x],'output_5/'+str(name[x][:-4])+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_gt.jpg',normalize=True)
            pred = pred.type(torch.FloatTensor)
            vutils.save_image(pred.data[x],'output_5/'+str(name[x][:-4])+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_res.jpg',normalize=True)


'''
for i, data in enumerate(dataloader, 1):
    input, gt,name = Variable(data[0]), Variable(data[1]),data[2] # [24, 1, 96, 96])
    name = np.array(name)
    # print(name.shape)
    # print(name[0])
    gt = gt.type(torch.FloatTensor)
    if cuda:
        input = input.cuda()
        target = gt.cuda()

    # print(type(input))
    # print(type(target))
    # print(isinstance(input,torch.FloatTensor))
    # print(isinstance(target,torch.FloatTensor))
    pred = NetS(input)
    pred = F.sigmoid(pred)

    input_mask = input.clone()
    output_masked = input_mask[:,0,:,:].unsqueeze(1) * pred
    if cuda:
        output_masked = output_masked.cuda()
    result = NetC(output_masked) 

    target_masked = input.clone()
    target_masked = input_mask[:,0,:,:].unsqueeze(1) * target
    if cuda:
        target_masked = target_masked.cuda()
    result_gt = NetC(target_masked)
    
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1


    pred = pred.type(torch.LongTensor)
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()

    # pred = pred.type(torch.LongTensor)
    # pred_np = pred.data.cpu().numpy()
    # gt = gt.data.cpu().numpy()
    result = result.cpu()
    result_gt = result_gt.cpu()
    for x in range(input.size()[0]):
        IoU = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
        dice = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
        IoUs.append(IoU)
        dices.append(dice)

        loss_D =  torch.mean(torch.abs(result[x] - result_gt[x])).detach().numpy()
        # print(loss_D)

        vutils.save_image(data[0][x],'output_test/'+str(name[x][:-4])+'_LoG_'+str((np.around(loss_D,decimals=4)))+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_.jpg',normalize=True)
        vutils.save_image(data[1][x],'output_test/'+str(name[x][:-4])+'_LoG_'+str((np.around(loss_D,decimals=4)))+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_gt.jpg',normalize=True)
        pred = pred.type(torch.FloatTensor)
        vutils.save_image(pred.data[x],'output_test/'+str(name[x][:-4])+'_LoG_'+str((np.around(loss_D,decimals=4)))+'_IoU_'+str(int(round(IoU,4)*10000))+'_dice_'+str(int(round(dice,4)*10000))+'_res.jpg',normalize=True)

    
    
    # break




IoUs = np.array(IoUs, dtype=np.float64)
dices = np.array(dices, dtype=np.float64)
mIoU = np.mean(IoUs, axis=0)
mdice = np.mean(dices, axis=0)
print('mIoU: {:.4f}'.format(mIoU))
print('Dice: {:.4f}'.format(mdice))
'''



