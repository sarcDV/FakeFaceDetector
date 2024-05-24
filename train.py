import os, sys
# to handle the sub-foldered structure of the executors
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, os.getcwd()) 
sys.path.insert(0, './utils')

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random, math
from glob import glob
from tqdm import tqdm
import logging
from statistics import median
from utilities import tensorboard_classification
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
"""train of ResNet-18 (18n building blocks) making only use of histogram  relative to images"""
__author__ = "Alessandro Sciarra"
__copyright__ = "Copyright 2024, Alessandro Sciarra & DEAS S.p.A."
__credits__ = ["Alessandro Sciarra"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "alessandro.sciarra@deas.it"
__status__ = "Under Testing"
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
### --- tensorboard --logdir TBLogs/ -------------------- ###
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
device = torch.device("cuda:0") # if torch.cuda.is_available() and cuda else "cpu")
log_path = './TBLogs'
trainID = 'RN18-Classification-BCELoss'
save_path = './results'
tb_writer = SummaryWriter(log_dir = os.path.join(log_path,trainID))
os.makedirs(save_path, exist_ok=True)
logname = os.path.join(save_path, 'log_'+trainID+'.txt')

logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
torch.manual_seed(0)

batch_size_ = 200
channels = 1
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
### --- Data Loader ------------------------------------- ###
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_datasets(root_dir, transform):
    datasets_dict = {'train': [], 'validation': [], 'test': []}
    for dataset in os.listdir(root_dir):
        for split in ['train', 'validation', 'test']:
            path = os.path.join(root_dir, dataset, split)
            datasets_dict[split].append(datasets.ImageFolder(path, transform=transform))
    # Concatenate all the datasets for each split
    for split in ['train', 'validation', 'test']:
        datasets_dict[split] = ConcatDataset(datasets_dict[split])
    return datasets_dict

root_dir = 'data'  # Replace with your root directory
datasets_dict = load_datasets(root_dir, transform)

# Create data loaders
batch_size = 8  # Replace with your batch size
train_loader = DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasets_dict['validation'], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False)


# # Get a batch of data
# data, labels = next(iter(val_loader))

# # Print the labels
# print(labels, data.shape)
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
### --- Model ------------------------------------------- ###
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
num_classes = 1
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
model.to(device)

### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
### --- Configure training ------------------------------ ###
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
start_epoch = 0
num_epochs= 2000
learning_rate = 1e-3 # 3e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scaler = GradScaler(enabled=True)
log_freq = 2
save_freq = 1
checkpoint = ""## "./results/RN18-Classification-BCELoss.pth.tar"
loss_func = nn.BCELoss()
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
if checkpoint:
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'])
    optimizer.load_state_dict(chk['optimizer'])
    scaler.load_state_dict(chk['AMPScaler'])  
    best_loss = chk['best_loss']  
    start_epoch = chk['epoch'] + 1
else:
    start_epoch = 0
    best_loss = float('inf')

### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
### --- Training session -------------------------------- ###
### ----------------------------------------------------- ###
### ----------------------------------------------------- ###
for epoch in range(start_epoch, num_epochs):
    ### --- Train --- ###
    model.train()
    runningLoss = []
    train_loss = []
    print('Epoch '+ str(epoch)+ ': Train')
    for idx, (data, labels) in enumerate(tqdm(train_loader)):
        out_ = torch.unsqueeze(labels.to(device),1).float()
        inp_ = data.to(device)
        
        optimizer.zero_grad()

        pred = model(inp_)
        loss = loss_func(pred, out_)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(loss)
        runningLoss.append(loss)
        logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, idx, len(train_loader), loss))
        
        ##For tensorboard
        if idx % log_freq == 0:
            niter = epoch*len(train_loader)+idx
            tb_writer.add_scalar('Train/Loss', median(runningLoss), niter)
            tensorboard_classification(tb_writer, inp_[0,...], out_[0], pred[0], epoch, section='train')
            runningLoss = []
    
    if epoch % save_freq == 0:            
        checkpoint = {
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'AMPScaler': scaler.state_dict()         
                }
        torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
                
    tb_writer.add_scalar('Train/EpochLoss', median(train_loss), epoch)
    ### ------  validation ------  ####
    if val_loader:
        model.eval()
        with torch.no_grad():
            runningLoss = []
            val_loss = []
            runningAcc = []
            val_acc = []
            print('Epoch '+ str(epoch)+ ': Val')
            for i, (data, labels) in enumerate(tqdm(val_loader)):
                out_ = torch.unsqueeze(labels.to(device),1).float()
                inp_ = data.to(device)
        
                pred = model(inp_)
                loss = loss_func(pred, out_)
                
                val_loss.append(loss)
                runningLoss.append(loss)
                
                logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))

                #For tensorboard
                if i % log_freq == 0:
                    niter = epoch*len(val_loader)+i
                    tb_writer.add_scalar('Val/Loss', median(runningLoss), niter)
                    tensorboard_classification(tb_writer, inp_[0,...], out_[0], pred[0], epoch, section='validation')
                    runningLoss = []
                    runningAcc = []
                    
            if median(val_loss) < best_loss:
                best_loss = median(val_loss)
                checkpoint = {
                            'epoch': epoch,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'AMPScaler': scaler.state_dict()         
                        }
                torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
                
        tb_writer.add_scalar('Val/EpochLoss', median(val_loss), epoch)
    
# ### ----------------------------------------------------- ###
# ### ----------------------------------------------------- ###
# ### ----------------------------------------------------- ###   