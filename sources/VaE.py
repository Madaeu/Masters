import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

features = 16

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, 
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

epochs = args['epochs']
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

class linearVAE(nn.Module):
    def __init__(self):
        super(linearVAE, self).__init__()

        #encoder
        self.encoder1 = nn.Linear(in_features=784, out_features=512)
        self.encoder2 = nn.Linear(in_features=512, out_features=features*2)

        #decoder
        self.decoder1 = nn.Linear(in_features=features, out_features = 512)
        self.decoder2 = nn.Linear(in_features=512, out_features=784)

    def reparameterize(self, mu, log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)
        sample=mu+(eps*std)
        return sample

    def forward(self, x):
        #encode
        x = F.relu(self.encoder1(x))
        x = self.encoder2(x).view(-1, 2, features)

        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        z = self.reparameterize(mu, log_var)

        #decode
        x = F.relu(self.decoder1(z))
        reconstruction = torch.sigmoid(self.decoder2(x))
        return reconstruction, mu, log_var

model = linearVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        print(reconstruction)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], 
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                #save_image(both.cpu(), f"../outputs/output{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")