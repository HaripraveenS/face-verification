import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
from torchvision import models
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import re
import pickle

def initialize_celeba(CELEBA_IMAGES = 30000):
    # global output_file_path, output_low_level_path, df_attributes
    attributes_path = "./content/drive/MyDrive/CelebA/metadata/list_attr_celeba.csv"
    df_attributes = pd.read_csv(attributes_path) 
    removeImages = df_attributes.shape[0] - CELEBA_IMAGES
    drop_indices = np.random.choice(df_attributes.index[1:], removeImages, replace=False)
    df_attributes = df_attributes.drop(drop_indices)     

    return df_attributes

class celebAData(Dataset):
    def __init__(self,df_attributes,transform=None,train=True):
        super().__init__()
        self.df_attributes = df_attributes
        self.transform = transform
        self.train = train
        self.data_list = df_attributes['image_id'].tolist()
        # celeba_id = df['image_id'].tolist()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,item):
        # global output_file_path, output_low_level_path, df_attributes
        # print(item)
        img_idx = item
        imgname = self.data_list[item]
        # foldername = imgname[:-9]
        # imgpath = os.path.join('/content/drive/MyDrive/LFW/zipped/lfw',foldername,imgname)
        imgpath = os.path.join("./content/drive/MyDrive/CelebA/zipped/img_align_celeba/img_align_celeba", imgname)
        # print(imgpath)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgr = cv2.resize(img,(224,224))
        label = self.df_attributes.iloc[img_idx][2:]
        label = np.array(label)
        print(label)
        label = np.where(label < 0,0,1)

        if self.transform is not None:
            imgr = self.transform(imgr)
        if self.train:
          return {
              'img' : imgr,
              'label' : torch.tensor(label)
          }
        else:
          return {
              'img':imgr
          }

class ResnetModel(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


df_attributes = initialize_celeba(20000)
attributesConsidered = 40
df_attributes_short = df_attributes.iloc[:, : attributesConsidered + 1]
df_attributes_short = df_attributes.reset_index()
print("dataframe shaep", df_attributes_short.shape)
print("dataframefirst column" , df_attributes_short.iloc[:10, :2])

dataset = celebAData(df_attributes_short)
tem = dataset.__getitem__(2)
plt.imshow(tem['img'])
#print(tem["label"])

test_data = celebAData(df_attributes_short)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
#len(test_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


considered_attr = 40
attr_model = ResnetModel(considered_attr).to(device)
attr_model.load_state_dict(torch.load("./attr_net_75k_40.pth"))

tem = torch.rand((32,3,224,224)).to(device)
y = attr_model(tem)
#print(y.shape)

attr_model.eval()
from tqdm import tqdm
attr_preds = []
print("FINDING ATTR PREDS:")
with torch.no_grad():
    for images in tqdm(test_loader):
        data = images['img'].squeeze(0).to(device)
        outputs = attr_model(data)
        pr = outputs.detach().cpu().numpy()
        for i in pr:
          attr_preds.append(np.round(i))  

def get_resnet():
    model1 = models.resnet50(pretrained=True)
    for name, child in model1.named_children():
        for param in child.parameters():
            param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 36)
    )
    return model1
print("FINDING simile PREDS:")

# model = model1.to(device)
simile_model = get_resnet().to(device)
simile_model.load_state_dict(torch.load("./simile_model.pth"))

simile_model.eval()
from tqdm import tqdm
simile_preds = []
with torch.no_grad():
    for images in tqdm(test_loader):
        data = images['img'].squeeze(0).to(device)
        outputs = simile_model(data)
        _, predicted = torch.max(outputs.data, 1)
        pr = outputs.detach().cpu().numpy()
        # print(pr)
        for i in pr:
          simile_preds.append(i) 

print("saving outputs")

with open("./celeba_simile_prediction.pkl","wb") as f:
    pickle.dump(simile_preds,f)

with open("./celeba_attr_prediction.pkl","wb") as f:
    pickle.dump(attr_preds,f)
