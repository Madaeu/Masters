import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip, RandomRotation
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import math
import time
from torchvision.utils import make_grid
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import onnx
import transforms as transforms
#from onnx_tf.backend import prepare

#from nyu_dataloader import NYUDataset
from NYUv2DepthDataLoader import NYUv2Dataset

EPOCHS = 80
CODE_SIZE = 0

def train_transform(intens, depth):
    angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip
    transform = transforms.Compose([
        transforms.Rotate(angle)
    ])

    intens = transform(intens)
    depth = transform(depth)
    intens = ToTensor()(intens)
    depth = ToTensor()(depth)

    depth = RandomHorizontalFlip(0.5)(depth)
    intens = RandomHorizontalFlip(0.5)(intens)

    return intens, depth

def double_conv(inputC, ouputC):
    conv = nn.Sequential(
        nn.Conv2d(inputC, ouputC, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ouputC, ouputC, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
    )
    return conv

def load_file(path):
    img = Image.open(path)
    img = ToTensor()(img)
    return img

# def jacobian(y, x, create_graph=False):                                                               
#     jac = []                                                                                          
#     flat_y = y.reshape(-1)                                                                            
#     grad_y = torch.zeros_like(flat_y)                                                                 
#     for i in range(len(flat_y)):                                                                      
#         grad_y[i] = 1.                                                                                
#         grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
#         jac.append(grad_x.reshape(x.shape))                                                           
#         grad_y[i] = 0.
#         print(i)                                                                           
#     return torch.stack(jac).reshape(y.shape + x.shape)

def scale_depth3(image):
    size = image.size
    #image_values = image.histogram()
    image_values = np.array(image)
    average = np.mean(image_values)
    for i in range(len(image_values)-1):
        image_values[i] = average / (average + image_values[i])

    image_array = np.array(image_values, dtype=np.float32)
    image2 = Image.new("L", size)
    image2.putdata(image_array)
    return image2


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.uncertainty128 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.uncertainty64 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.uncertainty32 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

        #U-net downconv layers
        self.conv1 = double_conv(1, 16)
        self.conv2 = double_conv(16, 32)
        self.conv3 = double_conv(32, 64)
        self.conv4 = double_conv(64, 128)
        self.conv5 = double_conv(128, 256)
        self.conv6 = double_conv(256, 256)


        #U-net upconv layers
        self.up_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv1 = double_conv(256, 128)
    
        self.up_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv2 = double_conv(128, 64)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upconv3 = double_conv(64, 32)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.upconv4 = double_conv(32, 16)

        self.out =nn.ConvTranspose2d(16, 1, 2, 2) # 3 eller 1 output channel
        self.out2 = nn.Conv2d(16,1,3,1,1)

    
    def forward(self, x):
        #down
        x1 = self.conv1(x)
        x2 = self.maxpool2x2(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool2x2(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool2x2(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool2x2(x7)
        x9 = self.conv5(x8)
        x10 = self.maxpool2x2(x9)
        x11 = self.conv6(x10)

        #up
        y1 = self.up_conv1(x11)
        y2 = self.upconv1(torch.cat([x8, y1], 1))
        y3 = self.up_conv2(y2)
        y4 = self.upconv2(torch.cat([x6, y3], 1))

        uncertainty32 = self.uncertainty32(y4)

        y5 = self.up_conv3(y4)
        y6 = self.upconv3(torch.cat([x4, y5], 1))

        uncertainty64 = self.uncertainty64(y6)

        y7 = self.up_conv4(y6)
        y8 = self.upconv4(torch.cat([x2, y7], 1))

        uncertainty128 = self.uncertainty128(y8)

        out = self.out2(self.upsample(y8))

        return y8, y6, y4, y2, x11, out, uncertainty128, uncertainty64, uncertainty32

class VaE(nn.Module):
    #def __init__(self, cat1, cat2, cat3, cat4, cat5):
    def __init__(self):
        super(VaE, self).__init__()

        self.cat1 = {}
        self.cat2 = {}
        self.cat3 = {}
        self.cat4 = {}
        self.cat5 = {}

        self.relu = nn.ReLU(inplace=True)
        #encoder
        self.enc1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.enc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.enc3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.enc4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.enc5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512*6*8, CODE_SIZE)
        self.fc2 = nn.Linear(512*6*8, CODE_SIZE)

        #decoder
        self.dec1 = nn.Linear(CODE_SIZE, 512*6*8)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dec2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(256)

        self.dec31 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.dec32 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.dec41 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.dec42 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(64)

        self.dec51 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.dec52 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.dec61 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.dec62 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn10 = nn.BatchNorm2d(16)

        self.dec71 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1, stride=1)
        
        #self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.out = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2)

        #predict layers
        self.predict128 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.predict64 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.predict32 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     # q = torch.distributions.Normal(mu, std)
    #     # z = q.rsample()
    #     # return z
    #     eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     eps = Variable(eps)
    #     return eps.mul(std).add_(mu)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):

        e1 = self.bn1(self.enc1(x))
        e1 = self.relu(e1)
        e1 = torch.cat([self.cat1, e1], 1)

        e2 = self.bn2(self.enc2(e1))
        e2 = self.relu(e2)
        e2 = torch.cat([self.cat2, e2], 1)

        e3 = self.bn3(self.enc3(e2))
        e3 = self.relu(e3)
        e3 = torch.cat([self.cat3, e3], 1)

        e4 = self.bn4(self.enc4(e3))
        e4 = self.relu(e4)
        e4 = torch.cat([self.cat4, e4], 1)

        e5 = self.bn5(self.enc5(e4))
        e5 = self.relu(e5)
        e5 = torch.cat([self.cat5, e5], 1)

        e5 = e5.view(-1, 512*6*8)

        return self.fc1(e5), self.fc2(e5)

    def decode(self, z):
        d1 = self.dec1(z)
        d1 = d1.view(-1, 512, 6, 8)

        d2 = self.bn6(self.dec2(self.up(d1)))
        d22 = self.cat5*d2

        d2 = torch.cat([self.cat5, d2, d22], 1)
        d2 = self.dec31(d2)

        d3 = self.bn7(self.dec32(self.up(d2)))
        d33 = self.cat4*d3
        d3 = torch.cat([self.cat4, d3, d33], 1)
        d3 = self.dec41(d3)

        d4 = self.bn8(self.dec42(self.up(d3)))
        d44 = self.cat3*d4
        d4 = torch.cat([self.cat3, d4, d44], 1)
        d4 = self.dec51(d4)

        pred64 = self.predict32(d4)

        d5 = self.bn9(self.dec52(self.up(d4)))
        d55 = self.cat2*d5
        d5 = torch.cat([self.cat2, d5, d55], 1)
        d5 = self.dec61(d5)

        pred32 = self.predict64(d5)

        d6 = self.bn10(self.dec62(self.up(d5)))
        d66 = self.cat1*d6
        d6 = torch.cat([self.cat1, d6, d66], 1)
        d6 = self.dec71(d6)

        pred16 = self.predict128(d6)

        out = self.out(d6)
        #out = self.out(self.up(d6))

        return out, pred64, pred32, pred16

    def forward(self, x, cat1, cat2, cat3, cat4, cat5):
        mu, logvar = self.encode(x.view(-1, 1, 192, 256))
        z = self.reparametrize(mu, logvar)
        x_hat, p64, p32, p16 = self.decode(z)
        return x_hat, mu, logvar, p64, p32, p16, z

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.enc1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.enc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.enc3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.enc4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.enc5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512*6*8, CODE_SIZE)
        self.fc2 = nn.Linear(512*6*8, CODE_SIZE)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x, cat1, cat2, cat3, cat4, cat5):

        e1 = self.bn1(self.enc1(x))
        e1 = self.relu(e1)
        e1 = torch.cat([cat1, e1], 1)

        e2 = self.bn2(self.enc2(e1))
        e2 = self.relu(e2)
        e2 = torch.cat([cat2, e2], 1)

        e3 = self.bn3(self.enc3(e2))
        e3 = self.relu(e3)
        e3 = torch.cat([cat3, e3], 1)

        e4 = self.bn4(self.enc4(e3))
        e4 = self.relu(e4)
        e4 = torch.cat([cat4, e4], 1)

        e5 = self.bn5(self.enc5(e4))
        e5 = self.relu(e5)
        e5 = torch.cat([cat5, e5], 1)

        e5 = e5.view(-1, 512*6*8)

        return self.fc1(e5), self.fc2(e5)

    def forward(self, x, cat1, cat2, cat3, cat4, cat5):
        mu, logvar = self.encode(x.view(-1, 1, 192, 256), cat1, cat2, cat3, cat4, cat5)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar


class EnsampleModel(nn.Module):
    def __init__(self, VaE, UNET):
        super(EnsampleModel, self).__init__() 
        self.vae = VaE
        self.unet = UNET
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    @torch.jit.export
    def decodeZeroCode(self, image, zero):

        device = torch.device('cuda:0')
        c1,c2,c3,c4,c5,uncertainty, unc128, unc64, unc32 = self.unet(image)
        self.vae.cat1 = c1
        self.vae.cat2 = c2
        self.vae.cat3 = c3
        self.vae.cat4 = c4
        self.vae.cat5 = c5
        dPred, p64, p32, p16 = self.vae.decode(zero)

        return dPred, p32, p16, uncertainty, unc128, unc64

    def forward(self, image, imageD):
        c1,c2,c3,c4,c5,uncertainty, uncertainty128, uncertainty64, uncertainty32 = self.unet(image)

        self.vae.cat1 = c1
        self.vae.cat2 = c2
        self.vae.cat3 = c3
        self.vae.cat4 = c4
        self.vae.cat5 = c5

        x_hat, mu, logvar, p64, p32, p16, z = self.vae(imageD, c1, c2, c3, c4, c5)

        return uncertainty, x_hat, mu, logvar, uncertainty128, uncertainty64, uncertainty32, p16, p32, p64, z #,grad

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using GPU:',torch.cuda.get_device_name(0))

vae = VaE().to(device)
unet = UNET().to(device)
encoder = Encoder().to(device)
ensample = EnsampleModel(vae, unet).to(device)

doubleoptim = torch.optim.Adam(ensample.parameters(), lr=1e-4)#,  weight_decay=1e-5)
VaEoptimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)#, weight_decay=1e-5)
UnetOptimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)#, weight_decay=1e-5)
EncoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)#, weight_decay=1e-5)

#scheduler = torch.optim.lr_scheduler.StepLR(doubleoptim, 10, 0.1)

criterion = nn.L1Loss(reduction='none')

#datas = ScannetDataset()
#data1 = torch.utils.data.DataLoader(dataset=datas, batch_size=6, shuffle=True)

#nyudata = NYUDataset("/home/nicklas/Desktop/data/nyudepthv2/train/", type='train')
#train_loader = torch.utils.data.DataLoader(dataset=nyudata, batch_size=5, shuffle=True)


#train_loader2 = torch.utils.data.DataLoader(dataset=nyudata2, batch_size=5, shuffle=True)
#print(len(train_loader2)
# print("Found ", nyudata2.__len__(), " images in train folder")

# nyudata2 = NYUv2Dataset()
# train_loader3, val_loader3 = torch.utils.data.random_split(nyudata2, [45000, 2584])

# train_loader3 = torch.utils.data.DataLoader(dataset=train_loader3, batch_size=5, shuffle=True)
# val_loader3 = torch.utils.data.DataLoader(dataset=val_loader3, batch_size=5, shuffle=True)

nyutest = NYUv2Dataset()
nyuVal = NYUv2Dataset(type="val")

train_loader4 = torch.utils.data.DataLoader(dataset=nyutest, batch_size=1, shuffle=True)
val_loader4 = torch.utils.data.DataLoader(dataset=nyuVal, batch_size=1, shuffle=False)

# valid = NYUv2Dataset(type="val")
# val_loader2 = torch.utils.data.DataLoader(dataset=valid, batch_size=1, shuffle=True)
# print("Found ", valid.__len__(), " images in validation folder")

#val_nyudata = NYUDataset("/home/nicklas/Desktop/data/nyudepthv2/val/", type='val')
#val_loader = torch.utils.data.DataLoader(dataset=val_nyudata, batch_size=5, shuffle=True)

criterion2 = nn.MSELoss(reduction='none')

def latent_loss(mean, logvar):
    #print(mean.size(), logvar.size())
    kld_loss = -0.5 * torch.mean((1 + logvar - mean.pow(2) - logvar.exp())) # her var der et ,1 som sidste argument til torch.sum: torch.sum(kl, 1)
    return kld_loss

def loss_function(y_pred, y, uncertainty):
    rec = criterion(y_pred, y)
    s = uncertainty
    loss  = torch.mean((rec*torch.exp(-s)) + s)
    return loss

def aleatoric_loss(y_pred, y, uncertainty):
    loss1 = torch.mean(torch.exp(-uncertainty) * torch.square( (y_pred - y)))
    loss2 = torch.mean(uncertainty)
    loss = 0.5*(loss1+loss2)
    return loss

def loss_function2(y_pred, y, uncertainty):
    rec = criterion(y_pred, y)
    reg = 0.5*(uncertainty)
    return torch.sum((torch.pow(rec,2)/ 2*uncertainty) + reg)

def Unettrain(save=False):
    unet.train()
    i = 0
    for epoch in range(EPOCHS):
        for data in tqdm(data1):
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)
            UnetOptimizer.zero_grad()
            c1, c2, c3, c4, c5, unetout = unet(image)
            loss = aleatoric_loss(unetout, imageD)
            loss.backward()
            UnetOptimizer.step()
            i += 1
            if i%5 == 0:
                save_image(unetout, "/home/nicklas/Desktop/unetvar/newimg" + str(i) + ".png")
                print("loss: ", loss)
    if save:
        torch.save(unet, '/home/nicklas/Desktop/MasterResults/NetModels/UNETmodelbs6Dropout.pth')

def VaEtrain(save=False):
    vae.train()
    unetpre = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/UNETmodelbs622.pth')
    i = 0
    for epoch in range(EPOCHS):
        for data in tqdm(data1):
            if i == 1:
                break
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)
            VaEoptimizer.zero_grad()
            c1, c2, c3, c4, c5, unetout = unetpre(image)
            x_hat, mu, logvar = vae(imageD, c1, c2, c3, c4, c5)
            loss2 = loss_function(x_hat, imageD, mu, logvar)
            loss2.backward()
            #l = criterion(x_hat, imageD)
            #l.backward()
            VaEoptimizer.step()
            i += 1
            if i%5 == 0:
                save_image(x_hat, "/home/nicklas/Desktop/vaeoutputs/newimg" + str(i) + ".png")
    if save:
        torch.save(vae,'/home/nicklas/Desktop/MasterResults/NetModels/VaEmodelbs622.pth')
    #save_image(x_hat, "/home/nicklas/Desktop/outputs/img1.png")

def Encodertrain(save=False):
    BigNet = EnsampleModel(vae, unet)
    BigNet = torch.load('/home/nicklas/Desktop/MasterResults/CodeSize32/Netmodels/EnsembleCODE3239.pth')
    BigNet.eval()
    encoder.train()
    i = 0
    for epoch in range(EPOCHS):
        for data in tqdm(train_loader4):
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)
            EncoderOptimizer.zero_grad()

            c1,c2,c3,c4,c5,uncertainty, unc128, unc64, unc32 = BigNet.unet(image)
            z, mu, logvar = encoder(image, c1, c2, c3, c4, c5)
            BigNet.vae.cat1 = c1
            BigNet.vae.cat2 = c2
            BigNet.vae.cat3 = c3
            BigNet.vae.cat4 = c4
            BigNet.vae.cat5 = c5

            x_hat, p64, p32, p16 = BigNet.vae.decode(mu)

            kl = latent_loss(mu, logvar) * 100
            loss = torch.mean(criterion(x_hat, imageD)) + kl
            loss.backward()
            EncoderOptimizer.step()
            i+=1
            if i%5==0:
                writer12.add_scalar('Encoder_Train_Loss', loss, i)

        EncoderTest(encoder, epoch, BigNet)
    if save:
        torch.save(encoder,'/home/nicklas/Desktop/MasterResults/NetModels/Encoder/Encodermodelbs622.pth')

def EncoderTest(model, epoch, bigmodel):
    i=0
    model.eval()
    bigmodel.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader4):
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)

            c1,c2,c3,c4,c5,uncertainty, unc128, unc64, unc32 = bigmodel.unet(image)

            z, mu, logvar = model(image, c1, c2, c3, c4, c5)

            bigmodel.vae.cat1 = c1
            bigmodel.vae.cat2 = c2
            bigmodel.vae.cat3 = c3
            bigmodel.vae.cat4 = c4
            bigmodel.vae.cat5 = c5

            x_hat, p64, p32, p16 = bigmodel.vae.decode(z)


            loss = torch.sum(criterion(x_hat, imageD))
            running_loss += loss
           

            if i%3==0:
                samlet = torch.cat([x_hat, imageD])
                unc = torch.exp(uncertainty)
                save_image(samlet, "/home/nicklas/Desktop/EncoderTest/out/newimg" + str(epoch) + str(i) + ".png")
                save_image(unc, "/home/nicklas/Desktop/EncoderTest/uncert/img" + str(epoch) + str(i) + ".png")
                save_image(image, "/home/nicklas/Desktop/EncoderTest/GT/GTimg" + str(i) + ".png")
            
            if epoch==0:
                f = open("demofile3.txt", "a")
                f.write("(" + str(i) + "," + str(int(loss.item())) + ")")
                f.close()
            i+=1

        writer11.add_scalar('Encoder_Validation_Loss', running_loss/len(val_loader4), epoch)
        print(running_loss/len(val_loader4))




writer = SummaryWriter('/home/nicklas/Desktop/testrun/avg_train_loss', comment='Avg_Train_Loss')
writer2 = SummaryWriter('/home/nicklas/Desktop/testrun/train_uncertainty/256', comment='Train_Uncertainty256x192')
writer3 = SummaryWriter('/home/nicklas/Desktop/testrun/train_loss', comment='Train_Loss')
writer4 = SummaryWriter('/home/nicklas/Desktop/testrun/train_uncertainty/128', comment='Train_Uncertainty128x96')
writer5 = SummaryWriter('/home/nicklas/Desktop/testrun/train_uncertainty/64', comment='Train_Uncertainty64x48')
writer6 = SummaryWriter('/home/nicklas/Desktop/testrun/train_uncertainty/32', comment='Train_Uncertainty32x24')
writer7 = SummaryWriter('/home/nicklas/Desktop/testrun/avg_val_loss', comment='Avg_Validation_Loss')
writer8 = SummaryWriter('/home/nicklas/Desktop/testrun/KL_Loss', comment='KL_Loss')
writer9 = SummaryWriter('/home/nicklas/Desktop/testrun/Train_ReconLoss', comment='Train_Recon_Loss')
writer10 = SummaryWriter('/home/nicklas/Desktop/testrun/img', comment='UncertaintyImg')
writer11 = SummaryWriter('/home/nicklas/Desktop/testrun/encoderVal', comment='EncoderValidation')
writer12 = SummaryWriter('/home/nicklas/Desktop/testrun/encoderTrain', comment='EncoderTrain')

valcriterion = nn.L1Loss(reduction='none')

def test(model, epoch):
    model.eval()
    running_loss=0.0
    resize128 = torchvision.transforms.Resize((96,128))
    resize64 = torchvision.transforms.Resize((48,64))
    resize32 = torchvision.transforms.Resize((24,32))
    i = 0
    with torch.no_grad():
        for data in tqdm(val_loader4):
            image, imageD = data
            image = image.to(device)#.unsqueeze(0)
            imageD = imageD.to(device)#.unsqueeze(0)

            img128 = resize128(imageD) #FIND BEDSTE UPSAMPLING MODE
            img64 = resize64(img128)
            img32 = resize32(img64)

            uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32, z = model(image, imageD)

            # loss256 = loss_function(x_hat, imageD, uncertainty)
            # loss128 = loss_function(p128, img128, uncert128) * 4
            # loss64 = loss_function(p64, img64, uncert64) * 16
            # loss32 = loss_function(p32, img32, uncert32) * 64
            # kl = latent_loss(mu, logvar)*100

            loss = torch.sum(valcriterion(x_hat, imageD))
            running_loss+=loss
            i+=1

            if i%3==0:
                samlet = torch.cat([x_hat, imageD])
                unc = torch.exp(uncertainty)
                save_image(samlet, "/home/nicklas/Desktop/encoderoutput/newimg" + str(epoch) + str(i) + ".png")
                save_image(unc, "/home/nicklas/Desktop/unetmean/256/img" + str(epoch) + str(i) + ".png")
                samlet2 = torch.cat([unc, image])
                save_image(samlet2, "/home/nicklas/Desktop/unetmean/256/img" + str(epoch) + str(i) + ".png")

            #running_loss += loss
       # print(running_loss/131)
        writer7.add_scalar('Avg_Validation_loss', running_loss/len(val_loader4), epoch)

        l = running_loss/len(val_loader4)
        
        f = open("testresults.txt", "a")
        f.write("(" + str(epoch) + "," + str(int(l.item())) + ")")
        f.close()


def train(save=False):
    i = 0
    resize128 = torchvision.transforms.Resize((96,128))
    resize64 = torchvision.transforms.Resize((48,64))
    resize32 = torchvision.transforms.Resize((24,32))
    for epoch in range(EPOCHS):
        running_loss = 0.0
        ensample.train()
        for data in tqdm(train_loader4):
            image, imageD = data
            image = image.to(device)#.unsqueeze(0)
            imageD = imageD.to(device)#.unsqueeze(0)
            doubleoptim.zero_grad()
            # img128 = F.interpolate(imageD, scale_factor=0.5, mode='bilinear') #FIND BEDSTE UPSAMPLING MODE
            # img64 = F.interpolate(img128, scale_factor=0.5, mode='bilinear')
            # img32 = F.interpolate(img64, scale_factor=0.5, mode='bilinear')

            img128 = resize128(imageD) #FIND BEDSTE UPSAMPLING MODE
            img64 = resize64(img128)
            img32 = resize32(img64)

            uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32, z = ensample(image, imageD)
            loss256 = loss_function(x_hat, imageD, uncertainty)
            loss128 = loss_function(p128, img128, uncert128) * 4
            loss64 = loss_function(p64, img64, uncert64) * 16
            loss32 = loss_function(p32, img32, uncert32) * 64

            kl = latent_loss(mu, logvar) * 100
            loss = loss256+loss128+loss64+loss32+kl
            running_loss += loss
            reconloss = torch.sum(valcriterion(x_hat, imageD))
            loss.backward()
            doubleoptim.step()
            i+=1
            #if i==1:
               #if torch.sum(uncertainty) < 0:
                    #return
            if i%5==0:
                writer3.add_scalar('Loss', loss, i)
                writer2.add_scalar('Uncertainty256x192', torch.mean(torch.exp(uncertainty)), i)
                writer4.add_scalar('Uncertainty128x96', torch.mean(torch.exp(uncert128)), i)
                writer5.add_scalar('Uncertainty64x48', torch.mean(torch.exp(uncert64)), i)
                writer6.add_scalar('Uncertainty32x24', torch.mean(torch.exp(uncert32)), i)
                writer8.add_scalar('KL_Loss', kl, i)
                writer9.add_scalar('Recon_Loss', reconloss, i)
            if i%500==0:
                unc = torch.exp(uncertainty)
                save_image(unc, "/home/nicklas/Desktop/unetmean/1/img" + str(epoch) + str(i) + ".png")
                save_image(image, "/home/nicklas/Desktop/GroundTruth/GTimg" + str(i) + ".png")
            #save_image(unc, "/home/nicklas/Desktop/unetmean/1/img" + str(epoch) + str(i) + ".png")
            #save_image(image, "/home/nicklas/Desktop/GroundTruth/GTimg" + str(i) + ".png")

            if epoch == EPOCHS-1:
                if i%5==0:
                    save_image(x_hat, "/home/nicklas/Desktop/vaeoutputs/newimg" + str(i) + ".png")
                    save_image(imageD, "/home/nicklas/Desktop/GroundTruth/GTimg" + str(i) + ".png")
        writer.add_scalar('Avg_Loss', running_loss/len(train_loader4), epoch)
    
        test(ensample, epoch)

        #if epoch < 15:
            #scheduler.step()
        if save:
            torch.save(ensample,'/home/nicklas/Desktop/MasterResults/NetModels/EnsembleCODE32' + str(epoch)+ '.pth')


def save_trace():
    model = EnsampleModel(vae, unet)
    model = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/EnsembleCODE3211.pth')

    device2 = torch.device("cpu")
    model = model.to(device2)
    example = torch.rand(1,1,192,256).to(device2)
    example2 = torch.rand(1,1,192,256).to(device2)
    model.eval()
    traced_script_module = torch.jit.trace(model, (example, example2))
    traced_script_module.save("/home/nicklas/Desktop/MasterResults/NetModels/SerializedModules/traced_model.pt")

def testModels():
    for i in range(40):
        model = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/EnsembleBS5NYUnew' + str(i)+ '.pth')
        print('/home/nicklas/Desktop/MasterResults/NetModels/EnsembleBS5NYUnew' + str(i)+ '.pth')
        test(model, i)


def decode_zero_code():
    model = EnsampleModel(vae, unet)
    model = torch.load('/home/nicklas/Desktop/MasterResults/CodeSize32/Netmodels/EnsembleCODE3239.pth')
    running_loss = 0.0
    model.eval()
    i = 0
    with torch.no_grad():
        for epoch in range(10):
            running_loss=0.0
            for data in tqdm(val_loader4):
                image, imageD = data
                imageD = imageD.to(device)
                image = image.to(device)
                #imageD = imageD.to(device)
                uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32, z = model(image, imageD)
                zero = torch.zeros((1,32)).to(device)
            # if i==255:
                #    zero = torch.zeros((1,32)).to(device)
                dPred, p32, p16, unc, unc128, unc64 = model.decodeZeroCode(image, zero)
                loss = torch.sum(criterion(dPred, imageD))
                running_loss+=loss


            print(running_loss/len(val_loader4))

def testmodelencoder():
    model = EnsampleModel(vae, unet)
    model = torch.load('/home/nicklas/Desktop/MasterResults/CodeSize32/Netmodels/EnsembleCODE3239.pth')
    model.eval()
    enc = Encoder()
    enc = torch.load('/home/nicklas/Desktop/MasterResults/SeperateEncoder/Encodermodelbs622.pth')
    enc.eval()
    running_loss = 0.0
    i = 0
    with torch.no_grad():
        for epoch in range(10):
            running_loss=0.0
            for data in tqdm(val_loader4):
                image, imageD = data
                imageD = imageD.to(device)
                image = image.to(device)

                c1,c2,c3,c4,c5,uncertainty, unc128, unc64, unc32 = model.unet(image)
                z, mu, logvar = encoder(image, c1, c2, c3, c4, c5)
                model.vae.cat1 = c1
                model.vae.cat2 = c2
                model.vae.cat3 = c3
                model.vae.cat4 = c4
                model.vae.cat5 = c5

                x_hat, p64, p32, p16 = model.vae.decode(mu)

                loss = torch.sum(criterion(x_hat, imageD)) + 200
                running_loss+=loss
                i+=1

            ting = running_loss/len(val_loader4)
            f = open("testresults2.txt", "a")
            f.write("(" + str(epoch) + "," + str(int(ting.item())) + ")")
            f.close()
        print(running_loss/len(val_loader4))


def decode_zero_code_image():
    model = EnsampleModel(vae, unet)
    model = torch.load('/home/nicklas/Desktop/MasterResults/CodeSize32/Netmodels/EnsembleCODE3239.pth')
    model.eval()
    resize = torchvision.transforms.Resize((192,256))
    with torch.no_grad():
        image = Image.open("/home/nicklas/Desktop/rgbd_dataset_freiburg3_structure_texture_far/rgb/1341839133.842375.png").convert("L")
        image = ToTensor()(image).unsqueeze(0)
        image = image.to(device)
        image = resize(image)
        

        #imageD = Image.open("/home/nicklas/Desktop/nyudata/newval/depth/img32078.png").convert("L")
        #imageD = ToTensor()(imageD).unsqueeze(0)
        #imageD = imageD.to(device)
        #uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32, z = model(image, imageD)
        zero = torch.zeros((1,32)).to(device)
        dPred, p32, p16, unc, unc128, unc64 = model.decodeZeroCode(image, zero)

        #sub = x_hat - dPred
        #sub = torch.mean(sub)
        #print(sub)
        #save_image(sub, "/home/nicklas/Desktop/subimg.png")

        save_image(dPred, "/home/nicklas/Desktop/zeroimg.png")
        unc = torch.exp(unc)
        save_image(unc, "/home/nicklas/Desktop/zerouncimg.png")
        save_image(image, "/home/nicklas/Desktop/intimg.png")

def evaluate(output, target):

    abs_diff = (output - target).abs()

    lg10 = float((log10(output) - log10(target)).abs().mean())

    #mse = float((torch.pow(abs_diff, 2)).mean())
    mse = (target - output)**2
    mse = mse.mean()
    rmse = math.sqrt(mse)

    maxRatio = torch.max(output / target, target / output)
    delta1 = float((maxRatio < 1.25).float().mean())

    return mse, rmse, delta1, abs_diff, lg10

def log10(x):
      """Convert a new tensor with the base-10 logarithm of the elements of x. """
      return torch.log(x) / math.log(10)

def val():
    model = EnsampleModel(vae, unet)
    i = 0
    sum_mse = 0.0
    sum_rmse = 0.0
    sum_delta = 0.0
    sum_abs = 0.0
    sum_lg = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    model = torch.load("/home/nicklas/Desktop/MasterResults/CodeSize32/Netmodels/EnsembleCODE3239.pth")
    with torch.no_grad():
        for data in tqdm(val_loader4):
            image, imageD = data
            imageD = imageD.to(device)
            image = image.to(device)
            uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32, z = model(image, imageD)
            zero = torch.zeros((1,32)).to(device)
            #start.record()
            dPred, p32, p16, unc, unc128, unc64 = model.decodeZeroCode(image, zero)
            #end.record()
            #torch.cuda.synchronize()
            #print(start.elapsed_time(end))
            #break
            save_image(dPred, "/home/nicklas/Desktop/tumseq3data/predictions/hat/" + str(i) + ".png")
            save_image(image, "/home/nicklas/Desktop/tumseq3data/predictions/im/" + str(i) + ".png")
            save_image(imageD, "/home/nicklas/Desktop/tumseq3data/predictions/gt/" + str(i) + ".png")
            i += 1
            mse, rmse, delta1, abs_diff, lg10 = evaluate(dPred, imageD)
            
            sum_mse += mse
            sum_rmse += rmse
            sum_delta += delta1
            sum_abs += abs_diff.mean()
            sum_lg += lg10

        avg_mse = sum_mse/len(val_loader4)
        avg_rmse = sum_rmse/len(val_loader4)
        avg_delta = sum_delta/len(val_loader4)
        avg_abs = sum_abs/len(val_loader4)
        avg_lg = sum_lg/len(val_loader4)

        print(avg_mse, avg_rmse, avg_delta, avg_abs, avg_lg)




#decode_zero_code()
#Encodertrain()
#decode_zero_code()
#train(True)
decode_zero_code_image()

#val()
#decode_zero_code()
#testmodelencoder()
#decode_zero_code()
#train(True)
#decode_zero_code()
# trained_model.load_state_dict(torch.load('/home/nicklas/Desktop/MasterResults/NetModels/EnsembleBS5NYUnew39.pth'))
# trained_model.eval()
# imageD = torch.rand(1,1,192,256).to(device)#
# torch.onnx.export(trained_model, (image,imageD), "/home/nicklas/Desktop/MasterResults/NetModels/model2.onnx", opset_version=11)

#model = onnx.load("/home/nicklas/Desktop/MasterResults/NetModels/model.onnx")
#decode_zero_code()

#NOTES
#Removed RELU from all prediction layers
#Added scheduler
#Added data transforms in NYU_Dataloader REMOVED
#Changed ReLu to sigmoid in uncertanty predictions REMOVED
#Added proximity
#Changed reparameterize trick to not use cuda floats
#added torch.exp to stuff with uncert
#changed reduce sum to means
#Using val folder for validation images doesnt work due to white outline on test images not being present in val images
