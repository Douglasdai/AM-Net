# CUDA_VISIBLE_DEVICES=X python train.py --cuda --outpath ./outputs
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
from net import AttU_Net ,Att_cls
from LoadData import Dataset, loader, Dataset_val
from net_gan import NetC
import logging
import time
from logger import Logger
import glob



logger_train = Logger('./logs_tensorborad/train_{}'.format(str(int(time.time()))))
logger_test = Logger('./logs_tensorborad/test_{}'.format(str(int(time.time()))))


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


logger_name = 'log_info/exp_{}.log'.format(str(int(time.time())))
fd = open(logger_name, mode="w", encoding="utf-8")
fd.close()
# logger = get_logger('log/exp.log')
logger = get_logger(logger_name)

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
parser.add_argument('--outpath', default='./outputs_att_info', help='folder to output images and model checkpoints')
opt = parser.parse_args()
print(opt)

# custom weights initialization called on NetS1,Net and NetC0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input, target):
    num = input * target
    num = torch.sum(num, dim=2)  # keepdim=True
    num = torch.sum(num, dim=2)

    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2)) + 1e-6

    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batchsize
    return dice_total

def to_float_and_cuda(input):
    input = input.type(torch.FloatTensor)
    input = input.cuda()
    return input


cuda = opt.cuda
torch.cuda.set_device(opt.ngpu)
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
print('===> Building model')
# NetS = AttU_Net(ngpu=opt.ngpu)
NetS = AttU_Net()
APNets = Att_cls()
if cuda:
    NetS = NetS.cuda()
    
# setup optimizer
lr = opt.lr
decay = opt.decay
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
#TODO for the classfiy optimizer  decide the size
optimizers = []
for i in range(9):
    optimizer = optim.Adam([{"params": NetS.parameters()}],
                           lr=lr, betas=(opt.beta1, 0.999))
    optimizers.append(optimizer)


# load training data
# train_datas = glob.glob('../dataprepare/images_train/*.jpg')
# train_labels = glob.glob('../dataprepare/masks_train/*.jpg')

# test_datas = glob.glob('../dataprepare/images_test/*.jpg')
# test_labels = glob.glob('../dataprepare/masks_test/*.jpg')

# lungs_data = glob.glob('../dataprepare/lungs_198/*.jpg')
# mediastinums_data = glob.glob('../dataprepare/mediastinums_198/*.jpg')

# lungs_test = glob.glob('../dataprepare/lungs_test/*jpg')
# mediastinums_test = glob.glob('../dataprepare/mediastinums_test/*jpg')

#5  prove loading data 
train_datas =[]
train_labels=[]
lungs_data =[]
mediastinums_data =[]
path = ['../data_5pictures/images_','../data_5pictures/masks_','../data_5pictures/lungs_','../data_5pictures/med_']
seq = ['2','3','4','5',]
jpg = '/*.jpg'
for i in range(4):
    train_path = (path[0]+seq[i]+jpg)
    train_data = glob.glob(train_path)
    train_datas.append(train_data)

    label_path = (path[1]+seq[i]+jpg)
    label_data =  glob.glob(label_path)
    train_labels.append(label_data)

    lungs_path = (path[2]+seq[i]+jpg)
    lung_data = glob.glob(lungs_path)
    lungs_data.append(lung_data)

    med_path = (path[3]+seq[i]+jpg)
    med_data =glob.glob(med_path)
    mediastinums_data.append(med_data)

train_datas = sum(train_datas,[])
train_labels = sum(train_labels,[])
lungs_data = sum(lungs_data,[])
mediastinums_data = sum(mediastinums_data,[])
# # print(len(train_datas),len(train_labels),len(lungs_data),len(mediastinums_data))

# #loading test data
test_datas = glob.glob('../data_5pictures/images_1/*.jpg')
test_labels = glob.glob('../data_5pictures/masks_1/*.jpg')
lungs_test = glob.glob('../data_5pictures/lungs_1/*jpg')
mediastinums_test = glob.glob('../data_5pictures/med_1/*jpg')

print('data info------------------------------------------------')
print(len(train_datas),len(train_labels),len(test_datas),len(test_labels),len(lungs_data),len(mediastinums_data))
print(len(lungs_data),len(mediastinums_data),len(lungs_test),len(mediastinums_test),len(lungs_test),len(mediastinums_test))
dataloader = loader(Dataset(train_datas,train_labels,lungs_data,mediastinums_data), opt.batchSize)
dataloader_val = loader(Dataset_val(test_datas,test_labels,lungs_test,mediastinums_test), opt.batchSize)
# load testing data
# dataloader_val = loader(Dataset_val('../../node/dataset/dataset_96_random_info/'), opt.batchSize)



max_iou = 0
NetS.train()
#TODO classfiy loss 
loss_function = nn.CrossEntropyLoss()
def cross_loss(input, target):
    # input = input.cuda()
    return loss_function(input, torch.squeeze(target).long().cuda())

#l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

for epoch in range(opt.niter):
	# for training
    for i, data in enumerate(dataloader, 1):
        input, label,input_192,lungs,mediastinums = Variable(data[0]), Variable(data[1]), Variable(data[2]),Variable(data[3]),Variable(data[4])
        mali_t, text_t, spicu_t, lobu_t, margin_t, spher_t, calci_t, sub_t, meter_t = Variable(data[5]), Variable(data[6]), Variable(data[7]),Variable(data[8]),Variable(data[9]),Variable(data[10]), Variable(data[11]), Variable(data[12]),Variable(data[13])
        
     
        if cuda:
            input = input.cuda()
            target = label.cuda()
            input_192 = input_192.cuda()
            lungs =lungs.cuda()
            mediastinums = mediastinums.cuda()

            input = to_float_and_cuda(input)
            target = to_float_and_cuda(target)
            input_192 = to_float_and_cuda(input_192)
            lungs =to_float_and_cuda(lungs)
            mediastinums = to_float_and_cuda(mediastinums)

            mali_t = to_float_and_cuda(mali_t)
            text_t = to_float_and_cuda(text_t)
            spicu_t = to_float_and_cuda(spicu_t)
            lobu_t = to_float_and_cuda(lobu_t)
            margin_t = to_float_and_cuda(margin_t)
            spher_t = to_float_and_cuda(spher_t)
            calci_t = to_float_and_cuda(calci_t)
            sub_t = to_float_and_cuda(sub_t)
            meter_t = to_float_and_cuda(meter_t)

        # original
        #zero_grad()
        NetS.zero_grad()
        #   GAN
        # NetC0.zero_grad()
        # # two outputs
        output,text, spicu, lobu, margin, spher, calci, sub, meter, mali = NetS(input,lungs,mediastinums)
        output = F.sigmoid(output)
        # output2 = F.sigmoid(output2)
        #TODO classfiy loss
        loss_atts = []
        loss_mali = cross_loss(mali, mali_t)
        loss_atts.append(loss_mali)
        loss_text = cross_loss(text, text_t)
        loss_atts.append(loss_text)
        loss_spicu = cross_loss(spicu, spicu_t)
        loss_atts.append(loss_spicu)
        loss_lobu = cross_loss(lobu, lobu_t)
        loss_atts.append(loss_lobu)
        loss_margin = cross_loss(margin, margin_t)
        loss_atts.append(loss_margin)
        loss_spher = cross_loss(spher, spher_t)
        loss_atts.append(loss_spher)
        loss_calci = cross_loss(calci, calci_t)
        loss_atts.append(loss_calci)
        loss_sub = cross_loss(sub, sub_t)
        loss_atts.append(loss_sub)
        loss_meter = torch.mean(torch.abs(meter - meter_t))
        loss_atts.append(loss_meter)

        #dice_loss
        loss_dice = dice_loss(output, target)
        
        #TODO change the final loss
        loss_G_joint =loss_dice +loss_mali + loss_meter + loss_text + loss_spicu + \
            loss_lobu + loss_margin + loss_spher + loss_calci + loss_sub
        loss_G_joint.backward(retain_graph=True)
        optimizerG.step()
        for i in range(9):
            loss_atts[i].backward(retain_graph=True)
            optimizers[i].step()
                     
    logger.info("===> Epoch[{}]({}/{}): Batch Dice: {:.4f},Dice_loss:{:.4f},text:{:.4f},spicu:{:.4f},lobu:{:.4f},margin:{:.4f},spher:{:.4f},calci:{:.4f},sub:{:.4f},meter:{:.4f},mali:{:.4f}".format
                (epoch, i, len(dataloader)
                    ,1 - loss_dice.item(), loss_dice.item(),
                    loss_text.item(), loss_spicu.item(), loss_lobu.item(), loss_margin.item(),
                    loss_spher.item(), loss_calci.item(), loss_sub.item(), loss_meter.item(),loss_mali.item()
                    #,loss_polar
                    # ,loss_G.item()
                    # ,loss_reconstruct.item()
                    # , loss_att1.item()
                    # , loss_att2.item(), loss_att3.item(), loss_att4.item()
                    # , loss_att6.item(), loss_att7.item(), loss_att8.item()

                    ))

    info_train = {'dice': 1 - loss_dice.item()}  # loss_dice.item()
    for tag, value in info_train.items():
        logger_train.scalar_summary(tag, value, epoch)
  
    vutils.save_image(data[0],
                      '%s/input.png' % opt.outpath,
                      normalize=True)
    vutils.save_image(data[1],
                      '%s/label.png' % opt.outpath,
                      normalize=True)
    vutils.save_image(output.data,
                      '%s/result.png' % opt.outpath,
                      normalize=True)

   #TODO testing  change the net ,other none change
    # for testing
    with torch.no_grad():
        if epoch % 1 == 0:
            NetS.eval()
            IoUs, dices = [], []
            for i, data in enumerate(dataloader_val, 1):
                input, gt ,lung, med = Variable(data[0]), Variable(data[1]),Variable(data[2]) ,Variable(data[3])
                #mali_t, text_t, spicu_t, lobu_t, margin_t, spher_t, calci_t, sub_t, meter_t =Variable(data[4]), Variable(data[5]), Variable(data[6]),Variable(data[7]),Variable(data[8]),Variable(data[9]), Variable(data[10]), Variable(data[11]),Variable(data[12])
                # print('test',input.size())
                if cuda:
                    input = to_float_and_cuda(input)
                    gt = to_float_and_cuda(gt)
                    lung = to_float_and_cuda(lung)
                    med = to_float_and_cuda(med)
                    
                    # mali_t = to_float_and_cuda(mali_t)
                    # text_t = to_float_and_cuda(text_t)
                    # spicu_t = to_float_and_cuda(spicu_t)
                    # lobu_t = to_float_and_cuda(lobu_t)
                    # margin_t = to_float_and_cuda(margin_t)
                    # spher_t = to_float_and_cuda(spher_t)
                    # calci_t = to_float_and_cuda(calci_t)
                    # sub_t = to_float_and_cuda(sub_t)
                    # meter_t = to_float_and_cuda(meter_t)
                pred,text, spicu, lobu, margin, spher, calci, sub, meter, mali = NetS(input,lung,med)
                # pred = pred[0]
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                pred = pred.type(torch.LongTensor)
                pred_np = pred.data.cpu().numpy()
                gt = gt.data.cpu().numpy()
                for x in range(input.size()[0]):
                    IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) +
                                                                np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]) + 1e-6)
                    dice = np.sum(pred_np[x][gt[x] == 1]) * 2 / float(np.sum(pred_np[x]) + np.sum(gt[x]) + 1e-6)
                    IoUs.append(IoU)
                    dices.append(dice)

            NetS.train()

            IoUs = np.array(IoUs, dtype=np.float64)
            dices = np.array(dices, dtype=np.float64)
            mIoU = np.mean(IoUs, axis=0)
            mdice = np.mean(dices, axis=0)
            print('mIoU: {:.4f}'.format(mIoU))
            print('Dice: {:.4f}'.format(mdice))
            
            logger.info('mIoU: {:.4f},Dice: {:.4f}'.format(mIoU, mdice))
            # tensorboard
            info_test = {'dice_train': mdice, 'IoU_test': mIoU}  # loss_dice.item()
            for tag, value in info_test.items():
                logger_test.scalar_summary(tag, value, epoch)

            # logger.info('Dice: {:.4f}'.format(mdice))
            if mIoU > max_iou:
                max_iou = mIoU
                torch.save(NetS.state_dict(), '%s/NetS_epoch_%d.pth' % (opt.outpath, epoch))
                # torch.save(NetC0.state_dict(), '%s/NetC0_epoch_%d.pth' % (opt.outpath, epoch))
            vutils.save_image(data[0],
                            '%s/input_val.png' % opt.outpath,
                            normalize=True)
            vutils.save_image(data[1],
                            '%s/label_val.png' % opt.outpath,
                            normalize=True)
            pred = pred.type(torch.FloatTensor)
            vutils.save_image(pred.data,
                            '%s/result_val.png' % opt.outpath,
                            normalize=True)
