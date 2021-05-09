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
import logging
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "--model",
    #     "-m",
    #     default="MODEL.pth",
    #     metavar="FILE",
    #     help="Specify the file in which the model is stored",
    # )
    # parser.add_argument(
    #     "--input",
    #     "-i",
    #     metavar="INPUT",
    #     nargs="+",
    #     help="filenames of input images",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--output", "-o", metavar="INPUT", nargs="+", help="Filenames of ouput images"
    # )
    # parser.add_argument(
    #     "--viz",
    #     "-v",
    #     action="store_true",
    #     help="Visualize the images as they are processed",
    #     default=False,
    # )
    # parser.add_argument(
    #     "--no-save",
    #     "-n",
    #     action="store_true",
    #     help="Do not save the output masks",
    #     default=False,
    # )

    parser.add_argument(
        "--inputimg",
        "-i",
        type=int,
        help="Minimum probability value to consider a mask pixel white",
        default=30000,
    )
    parser.add_argument(
        "--attributes",
        "-a",
        type=int,
        help="Scale factor for the input images",
        default=0.5,
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Scale factor for the input images",
        default=5,
    )

    return parser.parse_args()

def initialize_celeba(CELEBA_IMAGES = 30000):
    # global output_file_path, output_low_level_path, df_attributes
    attributes_path = "./content/drive/MyDrive/CelebA/metadata/list_attr_celeba.csv"
    df_attributes = pd.read_csv(attributes_path) 
    removeImages = df_attributes.shape[0] - CELEBA_IMAGES
    drop_indices = np.random.choice(df_attributes.index[1:], removeImages, replace=False)
    df_attributes = df_attributes.drop(drop_indices)     
    print(df_attributes.shape)
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
        label = self.df_attributes.iloc[img_idx][1:]
        label = np.array(label)
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

def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)

#from tqdm.notebook import tqdm
def train_test(modeltype,df_attributes,erro='bce',optimizertype='Adam',batch_size=32, epochs=5, transforms=None, modelPath = "./attr_net_small.pth"):
    # full_dataset = CustomData(transforms,train_path,trainimgs)
    print("=========================================================================================")
    print("=========================================================================================")
    full_dataset = celebAData(df_attributes, transforms)
    train_size = int(0.8 * len(full_dataset)) 
    test_size = len(full_dataset) - train_size

    # batch = 32
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    attributesConsidered = df_attributes.shape[1]-1
    print("attributes considered:", attributesConsidered)
    logging.info("attributes considered:{}".format(attributesConsidered))

    if modeltype == 'alexnet':
        model = AlexNet(attributesConsidered).to(device)
    elif modeltype == 'resnet':
        model = ResnetModel(attributesConsidered)
        model = model.to(device)
    elif modeltype == 'mobilenet':
        model = MultiOutputModel().to(device)
    else:
        raise Exception("Enter a valid model type!")
    
    if optimizertype == 'Adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizertype == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0,
                            weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    elif optimizertype == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    else:
        raise Exception("Enter a valid optimizer type!")

    if erro == 'ce':
        error = nn.CrossEntropyLoss().to(device)
    elif erro == 'ms':
        error = nn.MSELoss().to(device)
    elif erro == 'bce':
        error = nn.BCELoss().to(device)

    valid_loss_min = np.Inf

    losses = {'train' : [] } 
    accuracies = {'train' : []} 
    dataloaders = {
        'train':train_loader,
        'test':test_loader
    }

    print("dataloaders loaded")
    logging.info("dataloaders loaded")
    for epoch in range(epochs):
        print('='*10)
        print("Epoch: {}".format(epoch))

        logging.info('='*10)
        logging.info("Epoch: {}".format(epoch))

        model.train()
        # for phase in ['train']:

        curr_loss = 0.0
        curr_acc = 0

        for dinputs in dataloaders["train"]:
            inputs = dinputs["img"].to(device)
            labels = dinputs["label"].to(device)

            outputs = model(inputs)

            loss = error(outputs, labels.type(torch.float))
            # print(loss)
            curr_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()

            # _, preds = torch.max(outputs, 1)
            preds = outputs

            # curr_acc = 0.0
            for i,o in enumerate(outputs):
                acc = pred_acc(torch.Tensor.cpu(labels[i]), torch.Tensor.cpu(o))
                curr_acc += acc

        curr_loss = curr_loss / len(dataloaders["train"].sampler)
        curr_acc = np.asarray(curr_acc,dtype=np.float32) / len(dataloaders["train"].sampler)
        
        losses["train"].append(curr_loss)
        accuracies["train"].append(curr_acc)
        # if epoch == epochs - 1:
        print("train" + ":")
        print('loss = {:.4f}     accuracy = {:.4f}'.format(curr_loss,curr_acc))

        logging.info("train: ")
        logging.info('loss = {:.4f}     accuracy = {:.4f}'.format(curr_loss,curr_acc))
            # print()
        # train_losses.append(train_loss)
        # valid_losses.append(valid_loss)

    '''
    test code here
    '''
    print("testing" + ":")
    logging.info("testing: ")
    model.eval()
    test_loss = 0.0
    test_acc = 0
    for dinputs in dataloaders["test"]:
        inputs = dinputs["img"].to(device)
        labels = dinputs["label"].to(device)

        outputs = model(inputs)

        loss = error(outputs, labels.type(torch.float))
        test_loss += loss.item() * inputs.size(0)

        # _, preds = torch.max(outputs, 1)
        preds = outputs

        for i,o in enumerate(outputs):
            acc = pred_acc(torch.Tensor.cpu(labels[i]), torch.Tensor.cpu(o))
            test_acc += acc

    test_loss = test_loss / len(dataloaders["test"].sampler)
    test_acc = np.asarray(test_acc,dtype=np.float32) / len(dataloaders["test"].sampler)

    print("loss: {:.4f}     accuracy: {:.4f}".format(test_loss, test_acc))
    logging.info("loss: {:.4f}     accuracy: {:.4f}".format(test_loss, test_acc))

    torch.save(model.state_dict(), os.path.join(modelPath))
    print("saved model at:",modelPath)
    logging.info("saved model at: {}".format(modelPath))
    
    return losses,accuracies  

def main(modelPath = "./attr_net_small.pth"):
    args = get_args()
    noImages = args.inputimg
    epochs = args.epochs
    df_attributes = initialize_celeba(noImages)
    logging.info("celebA initialized")
    attributesConsidered = args.attributes
    df_attributes_short = df_attributes.iloc[:, : attributesConsidered + 1]
    print("attributes:", attributesConsidered)
    print("noImages:", noImages)
    print("epochs:",epochs)
    print("device:", device)
    logging.info("device:{}".format(device))

    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.3166, 0.3947, 0.4725), (0.1755, 0.1720, 0.1657))
    ])
    loss,acc = train_test("resnet",df_attributes_short,erro='bce',transforms=transforms_train, epochs =epochs, modelPath = modelPath )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
logging.basicConfig(filename='log_attr_net.log', level = logging.INFO)
main(modelPath = "./attr_net_75k_40.pth")
