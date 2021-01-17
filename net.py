import numpy as np
import torch
import torch.nn as nn
from numpy.random import normal
from math import sqrt
import argparse

add_indim = 36
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.up(x)

#TODO classfiy
class Att_base(nn.Module):
    def __init__(self, out_dim, in_dim=add_indim):
        super(Att_base, self).__init__()
        self.fc1_1 = nn.Linear(in_dim, 64)
        self.ac1_1 = nn.ReLU(inplace=True)
        self.fc1_2 = nn.Linear(64, 10)
        self.ac1_2 = nn.ReLU(inplace=True)
        self.fc1_3 = nn.Linear(10, out_dim)
        self.ac1_3 = nn.Sigmoid()

    def forward(self, input):
        # input = input.view()
        fc1 = self.fc1_1(input)
        fc1_ac = self.ac1_1(fc1)
        fc2 = self.fc1_2(fc1_ac)
        fc2_ac = self.ac1_2(fc2)
        fc3 = self.fc1_3(fc2_ac)
        out = self.ac1_3(fc3)
        return out, fc1_ac, fc2_ac
# 3 chuang wei
class Att_cls(nn.Module):
    def __init__(self, in_dim=add_indim):
        super(Att_cls, self).__init__()
        self.at1 = Att_base(2)
        self.at2 = Att_base(2)
        self.at3 = Att_base(2)
        self.at4 = Att_base(2)
        self.at5 = Att_base(2)
        self.at6 = Att_base(2)
        self.at7 = Att_base(2)
        self.at8 = Att_base(1)
        self.at9 = Att_base(2)
        
        # self.fc1 = nn.Linear(in_dim, 64)
        # self.fc1_ac = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(64 * 9, 10)
        # self.fc2_ac = nn.ReLU(inplace=True)

        # # self.fc2 = nn.Linear(in_dim, 10)
        # # self.fc2_ac = nn.ReLU(inplace=True)
        # self.fc3 = nn.Linear(10 * 9, 2)
        # self.fc3_ac = nn.Sigmoid()

    def forward(self, input,lung,med):
        # input = input.view()
        spic, spic_f1, spic_f2 = self.at2(lung)
        text, text_f1, text_f2 = self.at1(lung)

        lobu, lobu_f1, lobu_f2 = self.at3(med)
        sphe, sphe_f1, sphe_f2 = self.at5(med)

        marg, marg_f1, marg_f2 = self.at4(input)       
        calc, calc_f1, calc_f2 = self.at6(input)
        sube, sube_f1, sube_f2 = self.at7(input)
        mete, mete_f1, mete_f2 = self.at8(input)
        mali, mali_f1, mali_f2 = self.at9(input)

        return text, spic, lobu, marg, sphe, calc, sube, mete,mali


class AttU_Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, scale_factor=1):
        super(AttU_Net, self).__init__()
        # sixth = 32 make 192x192 photo
        filters = np.array([64, 128, 256, 512, 1024,32,16])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2_1 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3_1 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4_1 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5_1 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Conv1_2 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2_2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3_2 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4_2 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5_2 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Conv1_3 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2_3 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3_3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4_3 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5_3 = conv_block(ch_in=filters[3], ch_out=filters[4])
        
        # 3 in 1 concat and upsample
        self.Up5 = up_conv(ch_in=filters[4]*3, ch_out=filters[3]*3)
        # self.Att5 = Attention_block(F_g=filters[3]*3, F_l=filters[3]*3, F_int=filters[2]*3)
        self.Up_conv5 = conv_block(ch_in=filters[4]*3, ch_out=filters[3]*3)

        self.Up4 = up_conv(ch_in=filters[3]*3, ch_out=filters[2]*3)
        # self.Att4 = Attention_block(F_g=filters[2]*3, F_l=filters[2]*3, F_int=filters[1]*3)
        self.Up_conv4 = conv_block(ch_in=filters[3]*3, ch_out=filters[2]*3)

        self.Up3 = up_conv(ch_in=filters[2]*3, ch_out=filters[1]*3)
        # self.Att3 = Attention_block(F_g=filters[1]*3, F_l=filters[1]*3, F_int=filters[0]*3)
        self.Up_conv3 = conv_block(ch_in=filters[2]*3, ch_out=filters[1]*3)

        self.Up2 = up_conv(ch_in=filters[1]*3, ch_out=filters[0]*3)
        # self.Att2 = Attention_block(F_g=filters[0]*3, F_l=filters[0]*3, F_int=filters[0]*3 // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1]*3, ch_out=filters[0]*3)
        
        self.Conv_lung_1x1 = nn.Conv2d(filters[4], n_classes, kernel_size=1, stride=1, padding=0)
        self.conv_med_1x1 = nn.Conv2d(filters[4], n_classes, kernel_size=1, stride=1, padding=0)
        self.conv_ori_1x1 = nn.Conv2d(filters[4], n_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1 = nn.Conv2d(filters[0]*3, n_classes, kernel_size=1, stride=1, padding=0)
        self.cls = Att_cls()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, y, z):
        # encoding path
        # ori
        x1 = self.Conv1_1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2_1(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3_1(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4_1(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5_1(x5)

        # lung chuang 
        y1 = self.Conv1_2(y)
        y2 = self.Maxpool(y1)
        y2 = self.Conv2_2(y2)
        y3 = self.Maxpool(y2)
        y3 = self.Conv3_2(y3)
        y4 = self.Maxpool(y3)
        y4 = self.Conv4_2(y4)
        y5 = self.Maxpool(y4)
        y5 = self.Conv5_2(y5)
        # zong ge chuang 
        z1 = self.Conv1_3(z)
        z2 = self.Maxpool(z1)
        z2 = self.Conv2_3(z2)
        z3 = self.Maxpool(z2)
        z3 = self.Conv3_3(z3)
        z4 = self.Maxpool(z3)
        z4 = self.Conv4_3(z4)
        z5 = self.Maxpool(z4)
        z5 = self.Conv5_3(z5)
        #concat xyz
        xy5  =  torch.cat((x5,y5),dim=1)
        xyz5 = torch.cat((xy5,z5),dim=1)
        
        xy4  = torch.cat((x4,y4),dim =1)
        xyz4 = torch.cat((xy4,z4),dim =1)

        xy3 = torch.cat((x3,y3),dim =1)
        xyz3 = torch.cat((xy3,z3),dim=1)

        xy2 = torch.cat((x2,y2),dim=1)
        xyz2 = torch.cat((xy2,z2),dim=1)

        xy = torch.cat((x1,y1),dim=1)
        xyz = torch.cat((xy,z1),dim = 1)

        # decoding + concat path
        d5 = self.Up5(xyz5)
        # x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((xyz4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((xyz3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((xyz2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((xyz, d2), dim=1)
        d2 = self.Up_conv2(d2)

        #TODO classfiy
        x5 = self.conv_ori_1x1(x5)
        y5 = self.Conv_lung_1x1(y5)
        z5 = self.conv_med_1x1(z5)
        #print("x5,y5,z5",x5.size(),y5.size(),z5.size())
        x5, y5, z5 = x5.view(-1, 36), y5.view(-1, 36), z5.view(-1, 36)

        text, spicu, lobu, margin, spher, calci, sub, meter, mali = self.cls(x5,y5,z5)


        d1 = self.Conv_1x1(d2)
        
        return d1,text, spicu, lobu, margin, spher, calci, sub, meter, mali