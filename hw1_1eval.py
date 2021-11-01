import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()
test_dir = args.img_dir
out_file = args.save_dir

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import os

os.makedirs('realval_dataset')
for ii in range(50):
  os.makedirs('realval_dataset/'+str(ii))

for l in os.listdir(test_dir):
  a=l.strip('.png')
  b,c=a.split('_')
  os.rename(test_dir+'/'+str(l),'realval_dataset/'+b+'/'+str(l))


train_tfm = transforms.Compose([
    
    transforms.Resize((224, 224)),
    
    
    transforms.RandomCrop(224, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.5),
    
    torchvision.transforms.RandomRotation((-180,180)),
    
    
    transforms.ToTensor(),
])


test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 64

R_valid_set = DatasetFolder('realval_dataset', loader=lambda x: Image.open(x), extensions="png", transform=test_tfm)


realvalid_loader = DataLoader(R_valid_set, batch_size=batch_size, shuffle=False)


densenet121=models.densenet121(pretrained=True)
densenet161=models.densenet161(pretrained=True)
densenet169=models.densenet169(pretrained=True)
densenet201=models.densenet201(pretrained=True)

class d121m(nn.Module):

    def __init__(self):
        super(d121m, self).__init__()
        self.net=nn.Sequential(
            densenet121,
            nn.Linear(1000,50))
    def forward(self, x):
      x=self.net(x)
      return x


class d161m(nn.Module):

    def __init__(self):
        super(d161m, self).__init__()
        self.net=nn.Sequential(
            densenet161,
            nn.Linear(1000,50))
    def forward(self, x):
      x=self.net(x)
      return x


class d169m(nn.Module):

    def __init__(self):
        super(d169m, self).__init__()
        self.net=nn.Sequential(
            densenet169,
            nn.Linear(1000,50))
    def forward(self, x):
      x=self.net(x)
      return x


class d201m(nn.Module):

    def __init__(self):
        super(d201m, self).__init__()
        self.net=nn.Sequential(
            densenet201,
            nn.Linear(1000,50))
    def forward(self, x):
      x=self.net(x)
      return x

"""## **Training**


"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = nn.CrossEntropyLoss()
n_epochs = 70




d121=torch.load('d121_best.pt')
d161=torch.load('d161_best.pt')
d169=torch.load('d169_best.pt')
d201=torch.load('d201_best.pt')

d121.eval()
d161.eval()
d169.eval()
d201.eval()




predictions = []

test_loss=[]
test_accs=[]



for batch in realvalid_loader:
    
    imgs, labels = batch


    with torch.no_grad():
        l121=d121(imgs.to(device))
        l161=d161(imgs.to(device))
        l169=d169(imgs.to(device))
        l201=d201(imgs.to(device))
    

    logits =torch.add(l121,l161)
    logits=torch.add(logits,l169)
    logits=torch.add(logits,l201)
    logits=torch.div(logits,4)


   
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    
    
    loss = criterion(logits, labels.to(device))
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        
    test_loss.append(loss.item())
    test_accs.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_accs) / len(test_accs)
print(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")
with open('print.txt','a') as f:
    f.write(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")



filenames=[]
for i in range(50):
  for j in range(450,500):
    filenames.append(str(i)+'_'+str(j)+'.png')

with open(out_file, "w") as f:
    f.write("image_id,label\n")
    for i, pred in  enumerate(predictions):
         f.write(f"{filenames[i]},{pred}\n")


         
