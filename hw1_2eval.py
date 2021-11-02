import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()
test_dir = args.img_dir
out_file = args.save_dir

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

import os


os.makedirs('realval_img')
os.makedirs('realval_label')



for k in os.listdir(test_dir):
  if '.jpg' in k:
    a=k.strip('.jpg')
    b,c=a.split('_')
    os.rename(test_dir+'/'+str(k),'realval_img/'+str(int(b))+'.jpg')

  elif '.png' in k:
    a=k.strip('.png')
    b,c=a.split('_')
    os.rename(test_dir+'/'+str(k),'realval_label/'+str(int(b))+'.png')




class SDS(data.Dataset):
  def __init__(self,img_path,label_path,transform=None):
    self.transform=transform
    self.img_path=img_path
    self.label_path=label_path

  def __len__(self):
    return len(os.listdir(self.img_path))

  def __getitem__(self, id):
    img=Image.open(self.img_path+'/'+str(id)+'.jpg')
    label=Image.open(self.label_path+'/'+str(id)+'.png')
    if self.transform!=None:
      img=self.transform(img)
      label=self.transform(label)
    return img,label

transform=transforms.Compose([transforms.ToTensor()])


realval_set=SDS('realval_img','realval_label',transform)

batch_size=5
n_workers=0

test_loader = DataLoader(realval_set, batch_size=batch_size, shuffle=False)


vgg16=models.vgg16(pretrained=False)
dlresnet50=models.segmentation.deeplabv3_resnet50(pretrained=False)
resnet101fcn=models.segmentation.fcn_resnet101(pretrained=False)
resnet50fcn=models.segmentation.fcn_resnet50(pretrained=False)

class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()
        self.vgg=vgg16.features
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.conv=nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,7,3,1,1)
            )
    def forward(self,x):
        x=self.vgg(x)
        x=self.avgpool(x)
        x=self.conv(x)
        return x
    
class deeplab(nn.Module):
    def __init__(self):
        super(deeplab, self).__init__()
        self.dl=dlresnet50
        self.conv=nn.Sequential(nn.Conv2d(21,7,3,1,1))
    def forward(self,x):
        x=self.dl(x)['out']
        x=self.conv(x)

        return x
class resnet101FCN(nn.Module):
    def __init__(self):
        super(resnet101FCN, self).__init__()
        self.fcn=resnet101fcn
        self.conv=nn.Sequential(nn.Conv2d(21,7,3,1,1))
    def forward(self,x):
        x=self.fcn(x)['out']
        x=self.conv(x)

        return x
class resnet50FCN(nn.Module):
    def __init__(self):
        super(resnet50FCN, self).__init__()
        self.fcn=resnet50fcn
        self.conv=nn.Sequential(nn.Conv2d(21,7,3,1,1))
    def forward(self,x):
        x=self.fcn(x)['out']
        x=self.conv(x)

        return x


def read_masks(files):
    masks = np.empty((files.shape[0], 512, 512))
    masks = (masks+(np.min(masks)*-1))
    masks = masks / np.max(masks) * 6
    for i, file in enumerate(files):
        mask = file.cpu().detach().numpy()
        mask = 4 * mask[0, :, :] + 2 * mask[1, :, :] + mask[2, :, :]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(Pred, Labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    pred=Pred.argmax(dim=1).cpu().numpy()
    labels=Labels.cpu().numpy()
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if tp_fp + tp_fn - tp==0:
            iou=1
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou


device = "cuda"
criterion = nn.CrossEntropyLoss()
n_epochs = 50

def add0(s):
  if len(s)<4:
    n=4-len(s)
    for i in range(n):
      s='0'+s
  return s


import matplotlib.pyplot as plt 
model1 = torch.load('deeplab101 bestmodel 1.pt')
model2 = torch.load('deeplab101 bestmodel 2.pt')
model3 = torch.load('deeplab50 bestmodel 1.pt')
model4 = torch.load('deeplab50 bestmodel 2.pt')
model5 = torch.load('resnet50fcn bestmodel.pt')
model6 = torch.load('resnet101fcn bestmodel.pt')


model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()
test_loss=[]
test_iou=[]
masks=[]

for batch in test_loader:

    inputs, labels = batch
    
    with torch.no_grad():
      l1 = model1(inputs.to(device))
      l2 = model2(inputs.to(device))
      l3 = model3(inputs.to(device))
      l4 = model4(inputs.to(device))
      l5 = model5(inputs.to(device))
      l6 = model6(inputs.to(device))

    l5=torch.mul(l5,0.8)
    l6=torch.mul(l6,0.8)
    logits=torch.add(l1,l2)
    logits=torch.add(logits,l3)
    logits=torch.add(logits,l4)
    logits=torch.add(logits,l5)
    logits=torch.add(logits,l6)
    labels = read_masks(labels)
    labels = torch.from_numpy(labels).long().to(device)
    loss = criterion(logits, labels)

    test_loss.append(loss.item())

    iou=mean_iou_score(logits,labels)
    test_iou.append(iou)

    masks.append(logits.argmax(dim=1).cpu().numpy())

test_loss=sum(test_loss)/len(test_loss)
test_iou=sum(test_iou)/len(test_iou)

print(f"[ test ] loss = {test_loss:.5f}, iou = {test_iou:.5f}")



for j, mask in enumerate(masks):
  imgs = np.zeros((mask.shape[0], 512, 512, 3))
  for index, i in enumerate(mask):
    imgs[index, i == 0, 2] = 1
    imgs[index, i == 2, 2] = 1
    imgs[index, i == 4, 2] = 1
    imgs[index, i == 5, 2] = 1

    imgs[index, i == 0, 1] = 1
    imgs[index, i == 1, 1] = 1
    imgs[index, i == 3, 1] = 1
    imgs[index, i == 5, 1] = 1

    imgs[index, i == 1, 0] = 1
    imgs[index, i == 2, 0] = 1
    imgs[index, i == 5, 0] = 1

    fn = add0(str(j*5+index))+'_mask.png'
    output_path = os.path.join(out_file, fn)
    plt.imsave(output_path, imgs[index])




