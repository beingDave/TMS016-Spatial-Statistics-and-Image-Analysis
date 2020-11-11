#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:17:06 2020

@author: samaya, mdave
"""

import math, csv, os, time
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig
from torchsummary import summary
from torchvision import transforms
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


###############################
#####      PARAMETERS     #####
###############################

# model file names
model_names = ['model_jaffe_high_val_acc.pt', 'model_jaffe_low_val_loss.pt']

# facial emotion categories
CATEGORIES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

gpu = torch.cuda.is_available()

if not gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# initialize variables
X_train, y_train, X_t, y_t = [], [], [], []

###############################
#####      LOAD DATA      #####
###############################


data_path = os.getcwd()
data_path = data_path+"\\Jaffe\\"
imgs = []

filenames = sorted(os.listdir(data_path))
d = [] # vector of classification labels

for img_name in filenames:
    image = plt.imread(data_path + img_name)
    image_resized = resize(image, (48, 48))
    imgs.append(image_resized)
    if img_name[0:2] == 'an':
        d.append(0)
    if img_name[0:2] == 'di':
        d.append(1)
    if img_name[0:2] == 'fe':
        d.append(2)
    if img_name[0:2] == 'ha':
        d.append(3)
    if img_name[0:2] == 'ne':
        d.append(6)
    if img_name[0:2] == 'sa':
        d.append(4)
    if img_name[0:2] == 'su':
        d.append(5)

imgs = np.asarray(imgs)
d = np.asarray(d)

#Shuffle the dataset
x, y = shuffle(imgs, d, random_state=2)
X_dev, X_t, y_dev, y_t = train_test_split(x, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2)

###############################
#####      HELPER FCN     #####
############################### 

def give_me(how_many,x,y):
    start = 0
    end = how_many
    while end <= len(y) + how_many:
        yield x[start:end],y[start:end]
        start += how_many
        end += how_many 

###############################
#####      Early Stop     #####
###############################

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'[SAVED]: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), model_names[1])
        self.val_loss_min = val_loss        


####################################
#####           Model          #####
####################################

class MNetwork(nn.Module):
    def __init__(self):
        super(MNetwork, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1)
        
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(64)
        self.BN4 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(in_features=64*3*3, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=7)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, t):
        #(1) Input Layer
        #print("ip: " + str(t.shape))
        t = t                                           
        
        #(2) Hidden Convolution Layer 1
        t = F.relu(self.BN1(self.conv1(t)))
        t = F.max_pool2d(t, kernel_size=2)
        
        #(3) Hidden Convolution Layer 2
        t = F.relu(self.BN2(self.conv2(t)))
        t = F.max_pool2d(t, kernel_size=2)
        
        #(4) Hidden Convolution Layer 3
        t = F.relu(self.BN3(self.conv3(t)))
        t = F.max_pool2d(t, kernel_size=2) 
        
        #(5) Hidden Convolution Layer 4
        t = F.relu(self.BN4(self.conv4(t)))
        t = F.max_pool2d(t, kernel_size=2)
        
        #print("ip2op: " + str(t.shape))
        t = t.view(t.size(0), -1)
        
        #(6) Linear Layer 1
        t = self.dropout(t)
        t = F.relu(self.fc1(t))
    
        #(8) Output Layer
        t = self.dropout(t)
        t = self.out(t)
        
        return t

####################################
#####    Training Parameters   #####
####################################

Model = MNetwork()
criterion = nn.CrossEntropyLoss()
if gpu == 1:
    device=torch.device(0)
    Model = Model.to(device)
    criterion = criterion.to(device)
# SGD
# ln_rate = 0.1
# ln_factor = 0.05
# optimizer = optim.SGD(Model.parameters(), momentum=0.1, weight_decay=0.001, nesterov=True, lr=ln_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=ln_factor, patience=1, verbose=True)

# ADAM
ln_rate = 0.001
ln_factor = 0.05
optimizer = optim.Adam(Model.parameters(), lr=ln_rate, weight_decay= 0.001)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=ln_factor, patience=1, verbose=True)
#########################################    
########      Model Training     ########
#########################################

n_epochs = 1000
batch_size = 32

train_acc = 0
valid_acc = 0

avg_train_losses = []
avg_val_losses = []

avg_train_acc = []
avg_val_acc = [] 

valid_acc_max =  0.0
valid_loss_min = 0.0

#early_stopping = EarlyStopping(patience=100, verbose=True, delta=ln_rate*ln_factor*0.05)
early_stopping = EarlyStopping(patience=100, verbose=True, delta=ln_rate*0.05)

for epoch in range(1,n_epochs + 1):
    
    #shuffling train set each epoch
    #X_train, y_train = shuffle(X_train, y_train)
    train_loss = 0.0
    valid_loss = 0.0

    ##################################    
    ########     TRAINING     ########
    ##################################
    train_pictures = give_me(batch_size,X_train,y_train)
    Model.train()
    total = 0
    correct = 0
    
    for data,target in train_pictures:
        if len(data) > 0:
            optimizer.zero_grad()
            data = torch.FloatTensor(data).view(-1,1,48,48)
            target = torch.LongTensor(target)
            
            if gpu == 1:
                data, target = data.to(device), target.to(device)
            
            output = Model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)   
            total += target.size(0)
            correct += (pred == target).sum().item()
                
    train_acc = 100 * correct / total
            
    ##################################    
    ########    VALIDATION    ########
    ##################################
    valid_pictures = give_me(batch_size,X_val,y_val)
    Model.eval()
    total = 0
    correct = 0
    
    for data,target in valid_pictures:
        if len(data) > 0:
            data = torch.FloatTensor(data).view(-1,1,48,48)
            target = torch.LongTensor(target)
            
            if gpu == 1:
                data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                output = Model(data)
                loss = criterion(output,target)
                valid_loss += loss.item()*data.size(0)
                _, pred = torch.max(output, 1)   
                total += target.size(0)
                correct += (pred == target).sum().item()
                
    valid_acc = 100 * correct / total
        
    # calculate average losses
    train_loss = train_loss/len(X_train)
    valid_loss = valid_loss/len(X_val)
    avg_train_losses.append(train_loss)
    avg_val_losses.append(valid_loss)
    avg_train_acc.append(train_acc)
    avg_val_acc.append(valid_acc)
        
    # save model if validation acc has increased
    if valid_acc > valid_acc_max:
        print('[SAVED]: Validation acc increased ({:.6f} --> {:.6f})'.format(valid_acc_max,valid_acc))
        torch.save(Model.state_dict(), model_names[0])
        valid_acc_max = valid_acc
  
    # save model if validation loss has decreased
    if valid_loss < valid_loss_min:
        print('[SAVED]: Validation loss decreased ({:.6f} --> {:.6f})'.format(valid_loss_min,valid_loss))
        torch.save(Model.state_dict(), model_names[1])
        valid_loss_min = valid_loss
        
    # Scheduler step
    # scheduler.step(valid_loss)
    
    # print training/validation statistics 
    print('\nEpoch: {}/{}\nTraining Accuracy: {:.3f}\tTraining Loss: {:.6f}\nValidation Accuracy: {:.3f}\tValidation Loss: {:.6f}'.format(epoch, n_epochs, train_acc, train_loss, valid_acc, valid_loss))
    
    early_stopping(valid_loss, Model)  
        
    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    
#####################################    
########    MODEL TESTING    ########
#####################################

for ii in range(2):
    print(model_names[ii])
    Model = MNetwork()
    checkpoint = torch.load(model_names[ii])
    Model.load_state_dict(checkpoint)
    Model = Model.to(device)
    y_true = []
    y_pred = []
    y_tr = []
    y_pr = []

    test_loss = 0.0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))

    batch_size = 32
    test_pictures = give_me(batch_size,X_t,y_t)
    Model.eval()
    
    for data, target in test_pictures:
        if len(data) > 0:
            data = torch.FloatTensor(data).view(-1,1,48,48)
            target = torch.LongTensor(target)
            
            if gpu == 1:
                data, target = data.to(device), target.to(device)

            output = Model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            
            if gpu == 1:
                target = target.to(torch.device("cpu"))
                pred = pred.to(torch.device("cpu"))
            
            y_true.append(target.numpy())
            y_pred.append(pred.numpy())
            
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(X_t)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(7):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                CATEGORIES[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (CATEGORIES[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %
    (100. * np.sum(class_correct) / np.sum(class_total), 
    np.sum(class_correct), np.sum(class_total)))
    
    for j in range(len(y_pred)):
        for k in range(len(y_pred[j])):
            y_tr.append(y_true[j][k])
            y_pr.append(y_pred[j][k])
    
    y_true = np.asarray(y_tr)
    y_true = y_true.reshape(len(y_t), )
    y_pred = np.asarray(y_pr)
    y_pred = y_pred.reshape(len(y_t), )
    fig = plt.figure(i)
    df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred), CATEGORIES, CATEGORIES)
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.4) # for label size
    snsfig = sns.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='Blues', fmt='g') # font size
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    figure = snsfig.get_figure()
    cm_file = "jaffe_ConfusionMatrix_"+model_names[ii]+"_.png"
    figure.savefig(cm_file, dpi=80)
    plt.close()
    
    print('\n\n------------------------------------------------------------------\n\n')

#####################################    
########     Disp Losses     ########
#####################################
    
plt.figure(figsize=(12,12))
plt.plot(avg_train_losses)
plt.plot(avg_val_losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss")
plt.legend(("train loss", "validation loss"))
plt.savefig("jaffe_AvgLoss.png")

#####################################    
########    Disp Accuracy    ########
#####################################

plt.figure(figsize=(12,12))
plt.plot(avg_train_acc)
plt.plot(avg_val_acc)
plt.xlabel("EPOCHS")
plt.ylabel("Accuracy")
plt.legend(("training accuracy", "validation accuracy"))
plt.savefig("jaffe_AvgAcc.png")


#####################################    
####    Sample Training Data     ####
#####################################

z = 10
fig = plt.figure(figsize=(8,8))
fig.tight_layout()
fig.dpi=80
ax1 = fig.add_subplot(2,3,1)
ax1.imshow(x[z,:,:], cmap='gray')
ax1.grid(None)
ax1.set_title(CATEGORIES[y[z]])
ax2 = fig.add_subplot(2,3,2)
ax2.grid(None)
ax2.imshow(x[z+50,:,:], cmap='gray')
ax2.set_title(CATEGORIES[y[z+50]])
ax3 = fig.add_subplot(2,3,3)
ax3.grid(None)
ax3.imshow(x[z+77,:,:], cmap='gray')
ax3.set_title(CATEGORIES[y[z+77]])
ax4 = fig.add_subplot(2,3,4)
ax4.grid(None)
ax4.imshow(x[z+100,:,:], cmap='gray')
ax4.set_title(CATEGORIES[y[z+100]])
ax5 = fig.add_subplot(2,3,5)
ax5.imshow(x[z+150,:,:], cmap='gray')
ax5.grid(None)
ax5.set_title(CATEGORIES[y[z+150]])
ax6 = fig.add_subplot(2,3,6)
ax6.imshow(x[z+202,:,:], cmap='gray')
ax6.grid(None)
ax6.set_title(CATEGORIES[y[z+202]])
fig.savefig("jaffe_SampleData.png")

summary(Model, (1,48,48))
print(optimizer)
print(Model)
