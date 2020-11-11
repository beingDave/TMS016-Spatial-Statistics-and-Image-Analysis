import torch, os, csv
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from skimage.transform import resize
from torchvision import transforms
import matplotlib.pyplot as plt

data_path = os.getcwd()
model_name = "trained_model_fer2013_low_val_loss.pt"
model_path = data_path+"\\"+model_name

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1)
        
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(64)
        self.BN4 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(in_features=64*2*2, out_features=256)
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
        #t = F.softmax(t, dim=1)
        
        return t

Model = Network()
checkpoint = torch.load(model_path)
Model.load_state_dict(checkpoint)

model_weights = []
conv_layers = []
model_children = list(Model.children())

counter = 0 
 
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

for weight, conv in zip(model_weights, conv_layers):
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig("FilterBank_Layer1_"+model_name+".png")
plt.show()

#####################################    
######     Disp Feature Map    ######
#####################################

X_t, y_t = [], []

data_path = os.getcwd()

def load_test_data():
    fer_data1 = pd.read_csv(data_path+'\\fer2013test.csv')
    for index, row in fer_data1.iterrows():
        try:
            pixels=np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
            img = pixels.reshape((48,48))
            X_t.append(img)
            y_t.append(row['emotion'])
        except Exception as e:
            pass
        
load_test_data()
X_t = np.array(X_t)/255
y_t = np.array(y_t)
 

img = X_t[100,:,:]
img = torch.FloatTensor(img).view(1,1,48,48)
img = img[:][:,:,6:-6,6:-6]

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    print(results[i-1].shape)
    print(conv_layers[i-1])
    results.append(conv_layers[i](results[-1]))
    
outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 32:
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig("layer_"+str(num_layer)+"_"+model_name+".png")
    plt.close()