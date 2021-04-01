import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torchsummary import summary

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

class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.rootC = "/home/nicklas/Desktop/0 (copy)/0/photo"
        self.rootD = "/home/nicklas/Desktop/0 (copy)/0/depth"
        self.path_tempC = os.path.join(self.rootC, '%s', '%s.%s')
        self.path_tempD = os.path.join(self.rootD, '%s', '%s.%s')
        self.imtype = 'jpg'
        self.length = len(glob(self.path_tempC%('img', '*', self.imtype)))
        self.train = train
        self.rgb_transform = Compose([Resize([192, 256]), ToTensor()])
        self.depth_transform = Compose([Resize([192, 256]), ToTensor()])
        
    def __getitem__(self, index):
        
        img = Image.open( self.path_tempC%('img',str(index),self.imtype) )
  
        if self.train:
            depth = Image.open( self.path_tempD%('img',str(index),'png') )
            scale_depth3(depth)
            img, depth = self.rgb_transform(img), self.depth_transform(depth)
        else:
            depth = Image.open( self.path_temp%('depth',str(index).zfill(5),'png') )
            img, depth = ToTensor()(img), ToTensor()(depth)

        return img, depth.float()/65536

    def __len__(self):
#         return 16 # for debug purpose
        return self.length

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #U-net downconv layers
        self.conv1 = double_conv(3, 16)
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

    
    def forward(self, x):
        self.dropout = nn.Dropout(0.1)
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
#uncertainty
        y5 = self.up_conv3(y4)
        y6 = self.upconv3(torch.cat([x4, y5], 1))
#uncertainty
        y7 = self.up_conv4(y6)
        y8 = self.upconv4(torch.cat([x2, y7], 1))
#uncertainty
        out = self.out(y8)
        #out = self.sigmoid(out)
        out = self.relu(out)
#uncertainty

        return y8, y6, y4, y2, x11, out

class VaE(nn.Module):
    #def __init__(self, cat1, cat2, cat3, cat4, cat5):
    def __init__(self):
        super(VaE, self).__init__()

        # self.cat1 = cat1
        # self.cat2 = cat2
        # self.cat3 = cat3
        # self.cat4 = cat4
        # self.cat5 = cat5
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

        self.fc1 = nn.Linear(512*6*8, 512)
        self.fc2 = nn.Linear(512*6*8, 512)

        #decoder
        self.dec1 = nn.Linear(512, 512*6*8)
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
        self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)

        #predict layers
        self.predict128 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.predict64 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.predict32 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

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

    def decode(self, z, cat1, cat2, cat3, cat4, cat5):
        d1 = self.dec1(z)
        d1 = d1.view(-1, 512, 6, 8)

        d2 = self.bn6(self.dec2(self.up(d1)))
        d22 = cat5*d2
        d2 = torch.cat([cat5, d2, d22], 1)
        d2 = self.dec31(d2)

        d3 = self.bn7(self.dec32(self.up(d2)))
        d33 = cat4*d3
        d3 = torch.cat([cat4, d3, d33], 1)
        d3 = self.dec41(d3)

        d4 = self.bn8(self.dec42(self.up(d3)))
        d44 = cat3*d4
        d4 = torch.cat([cat3, d4, d44], 1)
        d4 = self.dec51(d4)

        pred64 = self.relu(self.predict32(d4))

        d5 = self.bn9(self.dec52(self.up(d4)))
        d55 = cat2*d5
        d5 = torch.cat([cat2, d5, d55], 1)
        d5 = self.dec61(d5)

        pred32 = self.relu(self.predict64(d5))

        d6 = self.bn10(self.dec62(self.up(d5)))
        d66 = cat1*d6
        d6 = torch.cat([cat1, d6, d66], 1)
        d6 = self.dec71(d6)

        pred16 = self.relu(self.predict128(d6))

        out = self.out(self.up(d6))

        return out, pred64, pred32, pred16

    def forward(self, x, cat1, cat2, cat3, cat4, cat5):
        mu, logvar = self.encode(x.view(-1, 1, 192, 256), cat1, cat2, cat3, cat4, cat5)
        z = self.reparametrize(mu, logvar)
        x_hat, p64, p32, p16 = self.decode(z, cat1, cat2, cat3, cat4, cat5)
        return x_hat, mu, logvar, p64, p32, p16

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.enc1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.enc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.enc3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.enc4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.enc5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(512*6*8, 512)
        self.fc2 = nn.Linear(512*6*8, 512)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

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
        mu, logvar = self.encode(x.view(-1, 3, 192, 256), cat1, cat2, cat3, cat4, cat5)
        z = self.reparametrize(mu, logvar)
        return z


class EnsampleModel(nn.Module):
    def __init__(self, VaE, UNET):
        super(EnsampleModel, self).__init__() 
        self.vae = VaE
        self.unet = UNET
        self.relu = nn.ReLU(inplace=True)
        self.uncertainty128 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.uncertainty64 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.uncertainty32 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)



    def forward(self, image, imageD):

        c1,c2,c3,c4,c5,uncertainty = self.unet(image)

        uncertainty128 = self.relu(self.uncertainty128(c1))
        uncertainty64 = self.relu(self.uncertainty64(c2))
        uncertainty32 = self.relu(self.uncertainty32(c3))

        x_hat, mu, logvar, p64, p32, p16 = self.vae(imageD, c1, c2, c3, c4, c5)
        return uncertainty, x_hat, mu, logvar, uncertainty128, uncertainty64, uncertainty32, p16, p32, p64

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using GPU:',torch.cuda.get_device_name(0))

# x = torch.rand(6, 3, 192, 256).to(device)
# x2 = torch.rand(6, 1, 192, 256).to(device)
#print(x.size())

vae = VaE().to(device)
unet = UNET().to(device)
encoder = Encoder().to(device)
ensample = EnsampleModel(vae, unet).to(device)


doubleoptim = torch.optim.Adam(ensample.parameters(), lr=1e-4, betas=(0.9, 0.999))
VaEoptimizer = torch.optim.Adam(vae.parameters(), lr=1e-4, betas=(0.9, 0.999))
UnetOptimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, betas=(0.9, 0.999))
EncoderOptimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.9, 0.999))

criterion = nn.L1Loss(reduction='none')

datas = ScannetDataset()
data1 = torch.utils.data.DataLoader(dataset=datas, batch_size=6, shuffle=True)
EPOCHS = 5

def latent_loss(mean, logvar):
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) # her var der et ,1 som sidste argument til torch.sum: torch.sum(kl, 1)
    return kld_loss

def loss_function(y_pred, y, uncertainty):
    rec = criterion(y_pred, y)
    s = uncertainty
    loss  = torch.sum((rec*torch.exp(-s)) + s)
    return loss

def loss_function2(y_pred, y, uncertainty):
    rec = criterion(y_pred, y)
    reg = uncertainty
    return torch.sum(rec/ torch.exp(uncertainty) + reg)

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
    encoder.train()
    un = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/UNETmodelbs622.pth')
    varenc = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/VaEmodelbs622.pth')
    i = 0
    for epoch in range(EPOCHS):
        for data in tqdm(data1):
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)
            EncoderOptimizer.zero_grad()
            c1, c2, c3, c4, c5, unetout = un(image)
            z = encoder(image, c1, c2, c3, c4, c5)
            x_hat = varenc.decode(z, c1, c2, c3, c4, c5)
            loss = criterion(x_hat, imageD)
            loss.backward()
            EncoderOptimizer.step()
    

    if save:
        torch.save(encoder,'/home/nicklas/Desktop/MasterResults/NetModels/Encodermodelbs622.pth')

def train(save=False):
    unet.train()
    vae.train()
    i = 0
    for epoch in range(EPOCHS):
        for data in tqdm(data1):
            image, imageD = data
            image = image.to(device)
            imageD = imageD.to(device)
            doubleoptim.zero_grad()

            img128 = F.interpolate(imageD, scale_factor=0.5) #FIND BEDSTE UPSAMPLING MODE
            img64 = F.interpolate(img128, scale_factor=0.5)
            img32 = F.interpolate(img64, scale_factor=0.5)

            #c1,c2,c3,c4,c5,unetout = unet(image)
            #x_hat, mu, logvar = vae(imageD, c1, c2, c3, c4, c5)
            uncertainty, x_hat, mu, logvar, uncert128, uncert64, uncert32, p128, p64, p32 = ensample(image, imageD)

            loss128 = loss_function(p128, img128, uncert128) * 4
            loss64 = loss_function(p64, img64, uncert64) * 16
            loss32 = loss_function(p32, img32, uncert32) * 64

            kl = latent_loss(mu, logvar)
            loss = loss_function(x_hat, imageD, uncertainty)+kl+loss128+loss64+loss32
            loss.backward()
            doubleoptim.step()
            i+=1
            if i%5==0:
                print(loss)
                save_image(x_hat[0], "/home/nicklas/Desktop/vaeoutputs/newimg" + str(i) + ".png")
                save_image(uncertainty[0], "/home/nicklas/Desktop/unetmean/newimg" + str(i) + ".png")
    if save:
        torch.save(ensample,'/home/nicklas/Desktop/MasterResults/NetModels/EnsembleBS6.pth')
def unetval():
    unt = torch.load('/home/nicklas/Desktop/MasterResults/NetModels/UNETmodelbs6Dropout.pth')
    unt.eval()

    im, imD = datas.__getitem__(5)
    im = im.unsqueeze(0).to(device)
    imD = imD.unsqueeze(0).to(device)
    with torch.no_grad():
        c1, c2, c3, c4, c5, unetout = unt(im)
        loss = criterion(unetout, imD)
        print(loss)
        save_image(unetout, '/home/nicklas/Desktop/unetout.png')

valdata = ScannetDataset()
valimgs = torch.utils.data.DataLoader(dataset=valdata, batch_size=1, shuffle=True)

#Unettrain()
#VaEtrain()
#Encodertrain()
#unetval()
train()