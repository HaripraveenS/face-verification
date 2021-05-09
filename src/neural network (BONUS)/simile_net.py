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


def initialize_celeb_face_recog():
    global output_file_path, output_low_level_path, list_ref_persons, ref_file_path
    output_file_path = "./content/drive/MyDrive/project-sepnu/data/simile classifier/celeb_face_recog/"
    output_low_level_path = "./content/drive/MyDrive/project-sepnu/data/low level/celeb_face_recog/"
    ref_file_path = "./content/drive/MyDrive/CelebrityFaceRecognition/reference_faces/"
    file_ref_persons = open(ref_file_path + "reference_faces.txt", "r")
    list_ref_persons = [ref_person.replace("\n", "") for ref_person in file_ref_persons.readlines()]
    file_ref_persons.close()


class CelebRecogData(Dataset):
    def __init__(self,data_list,map_name_label,data_dir,transform=None,train=True):
        super().__init__()
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.map_name_label = map_name_label

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,item):
        img_idx = item
        imgname = self.data_list[item]
        celeb_name = imgname[0]
        img_num = imgname[1]
        imgpath = os.path.join(self.data_dir,celeb_name,img_num)
        #print(imgpath,item)
        img = cv2.imread(imgpath)
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except:
            print(imgpath,item)
        imgr = cv2.resize(img,(224,224))
        label = self.map_name_label[celeb_name]
        if self.transform is not None:
            imgr = self.transform(imgr)
        return {'img' : imgr,'label' : torch.tensor(label)}

initialize_celeb_face_recog()
# print(list_ref_persons)

path = "./content/drive/MyDrive/CelebrityFaceRecognition/zipped"
map_name_label = {list_ref_persons[i]:i for i in range(len(list_ref_persons))}
datalist = []
labellist = []
for name in list_ref_persons:
    name_path = os.path.join(path,name)
    print(name)
    name_img_list = os.listdir(name_path)
    for file_name in name_img_list:
        if file_name.endswith(".jpg"):
            datalist.append([name,file_name])
            labellist.append(map_name_label[name])
        # else :
            # print("exception occured!",name,file_name)

transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3166, 0.3947, 0.4725), (0.1755, 0.1720, 0.1657)),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:


print(device)

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
                nn.Linear(512, len(list_ref_persons)))
    return model1

from tqdm import tqdm
def train_test(data_list,data_dir,transforms=None):
    # full_dataset = CustomData(transforms,train_path,trainimgs)
    # print("=========================================================================================")
    # print("Current Attribute:", attr)
    full_dataset = CelebRecogData(datalist,map_name_label,path,transforms)

    train_size = int(0.8 * len(full_dataset)) 
    test_size = len(full_dataset) - train_size

    batch = 128
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    model = get_resnet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss().to(device)
    num_epochs = 50
    valid_loss_min = np.Inf

    losses = {'train' : [] } 
    accuracies = {'train' : []} 
    dataloaders = {
        'train':train_loader,
        'test':test_loader
    }

    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        curr_loss = 0.0
        curr_acc = 0

        for dinputs in tqdm(dataloaders["train"]):
            inputs = dinputs["img"].to(device)
            labels = dinputs["label"].to(device)

            outputs = model(inputs)
            loss = error(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            curr_loss += loss.item() * inputs.size(0)
            curr_acc += torch.sum(preds == labels.data)

        curr_loss = curr_loss / len(dataloaders["train"].sampler)
        curr_acc = curr_acc.double() / len(dataloaders["train"].sampler)
        
        losses["train"].append(curr_loss)
        accuracies["train"].append(curr_acc)
        # if epoch == num_epochs - 1:
        print("train" + ":")
        print('loss = {:.4f}     accuracy = {:.4f}'.format(curr_loss,curr_acc))
            # print()
        # train_losses.append(train_loss)
        # valid_losses.append(valid_loss)
        torch.save(model.state_dict(), os.path.join("simile_model.pth"))
        print("saved model at:","simile_model.pth")

    '''
    test code here
    '''
    model.eval()
    test_loss = 0.0
    test_acc = 0
    for dinputs in dataloaders["test"]:
        inputs = dinputs["img"].to(device)
        labels = dinputs["label"].to(device)

        outputs = model(inputs)
        loss = error(outputs, labels)

        _, preds = torch.max(outputs, 1)
        test_loss += loss.item() * inputs.size(0)
        test_acc += torch.sum(preds == labels.data)

    test_loss = test_loss / len(dataloaders["test"].sampler)
    test_acc = test_acc.double() / len(dataloaders["test"].sampler)
    print("test" + ":")
    print("loss: {:.4f}     accuracy: {:.4f}".format(test_loss, test_acc))

    return losses,accuracies

loss,acc = train_test(datalist,path,transforms_train)

