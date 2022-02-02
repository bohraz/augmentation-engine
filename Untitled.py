#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from torchvision import datasets, transforms, models
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from filament_augmentation.loader.filament_dataloader import FilamentDataLoader
from filament_augmentation.generator.filament_dataset import FilamentDataset
from filament_augmentation.metadata.filament_metadata import FilamentMetadata



bbso_path =  r'D:\GSU_Assignments\Summer_sem\RA\bbso_data_retriever\bbso_fulldisk'
json_path = r'D:\GSU_Assignments\Summer_sem\filaments_data_augmentation\chir_data.json'


# In[2]:



filamentInfo = FilamentMetadata(ann_file = json_path, start_time = '2000-01-01 00:00:00',
                                    end_time = '2016-12-31 23:59:59')
filamentInfo.get_chirality_distribution()


# In[3]:


transforms1 = [
    transforms.ColorJitter(brightness=(0.25,1.25), contrast=(0.25,2.00), saturation=(0.25,2.25)),
    transforms.RandomRotation(15,expand=False,fill=110)
]

train_dataset = FilamentDataset(bbso_path = bbso_path, ann_file = json_path, 
                          start_time = "2000-01-01 00:00:00", end_time = "2014-12-31 11:59:59")
validate_dataset = FilamentDataset(bbso_path = bbso_path, ann_file = json_path, 
                          start_time = "2015-01-01 00:00:00", end_time = "2015-12-31 11:59:59")
print(len(train_dataset.data))
train_data_loader = FilamentDataLoader(dataset = train_dataset,batch_size = 99 , filament_ratio = (1, 1, 1),n_batchs = 10, 
                                 transforms = transforms1, image_dim = 224)
validate_data_loader = FilamentDataLoader(dataset = validate_dataset,batch_size = 30 , filament_ratio = (1, 1, 1),n_batchs = 3, 
                                 transforms = transforms1, image_dim = 224)


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model1 = models.resnet50(pretrained=False)
for param in model1.parameters():
    param.requires_grad = False
    
model1.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model1.fc.parameters(), lr=0.007)
#SGD
model1.to(device)


# In[5]:


epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, validate_losses, train_acc, validate_acc = [], [], [], []
for epoch in range(epochs):
    for _,inputs, labels in train_data_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
#         print(inputs, labels)
        optimizer.zero_grad()
        logps = model1.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            validate_loss = 0
            accuracy = 0
            model1.eval()
            with torch.no_grad():
                for _,inputs, labels in validate_data_loader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model1.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    validate_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_data_loader))
            validate_losses.append(validate_loss/len(validate_data_loader))     
            validate_acc.append(accuracy/len(validate_data_loader))
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validate loss: {validate_loss/len(validate_data_loader):.3f}.. "
                  f"validate accuracy: {accuracy/len(validate_data_loader):.3f}")
            running_loss = 0
            model1.train()
torch.save(model1, 'smartsamplemodel.pth')


# In[6]:


import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(validate_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


# In[7]:


plt.plot(validate_acc, label='Train Accuracy')


# In[14]:


test_data = 'D:\GSU_Assignments\Semester_2\DL\evalutate_augmentation_engine\\test_images'


# In[15]:


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

class Filament_Dataset(Dataset):
    def __init__(self, img_dir, img_dim):
        self.img_dir = img_dir
        self.Rdata = os.listdir(os.path.abspath(os.path.join(img_dir,'R')))
        self.Ldata = os.listdir(os.path.abspath(os.path.join(img_dir,'L')))
        self.Udata = os.listdir(os.path.abspath(os.path.join(img_dir,'U')))
        self.data = self.Rdata + self.Ldata + self.Udata
        self.classes = [1]*len(self.Rdata) + [2]*len(self.Ldata) + [0]*len(self.Udata)
        self.img_dim = (img_dim, img_dim)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image = self.data[idx]
        class_name = self.classes[idx]
        if class_name == 1:
            image_path = os.path.abspath(os.path.join(self.img_dir,'R',image))
        elif class_name == 2:
            image_path = os.path.abspath(os.path.join(self.img_dir,'L',image))
        elif class_name == 0:
            image_path = os.path.abspath(os.path.join(self.img_dir,'U',image))
        image = Image.open(image_path) 
        image = image.convert('RGB')
        image = np.array(image, dtype = np.float32)
        
        if self.img_dim != 0:
            image = cv2.resize(image, self.img_dim)
        img_tensor = torch.from_numpy(image)
        img_tensor = torch.swapaxes(img_tensor, 2, 0)
        class_id = torch.tensor(class_name)
        return img_tensor, class_id


# In[16]:


test = Filament_Dataset(test_data, 256)


# In[20]:


testloader = DataLoader(test, batch_size=10, shuffle=True)


# In[21]:


predictions, actuals = list(), list()
for inputs, labels in testloader:
    x_test = model1(inputs)
    x_test = x_test.detach().numpy()
    actual = labels.numpy()
    actual = actual.reshape((len(actual),1))
    x_test = x_test.round()
    predictions.append(x_test)
    actuals.append(actual)
predictions, actuals = np.vstack(predictions), np.vstack(actuals)    


# In[22]:


rounded_labels=np.argmax(predictions, axis=1)
rounded_labels


# In[24]:


from sklearn.metrics import classification_report

print(classification_report(actuals , rounded_labels, target_names = ['Unidentified Chirality', 'Right Chirality', 'Left Chirality'] ))


# In[36]:


transforms1 = [
    transforms.ColorJitter(brightness=(0.25,1.25), contrast=(0.25,2.00), saturation=(0.25,2.25)),
    transforms.RandomRotation(15,expand=False,fill=110)
]

test_dataset = FilamentDataset(bbso_path = bbso_path, ann_file = json_path, 
                          start_time = "2016-01-01 00:00:00", end_time = "2016-12-31 11:59:59")
test_data_loader = FilamentDataLoader(dataset = test_dataset,batch_size = 120 , filament_ratio = (1, 1, 1),n_batchs = 5, 
                                 transforms = transforms1, image_dim = 224)


# In[37]:


predictions, actuals = list(), list()
for _,inputs, labels in test_data_loader:
    x_test = model1(inputs)
    x_test = x_test.detach().numpy()
    actual = labels.numpy()
    actual = actual.reshape((len(actual),1))
    x_test = x_test.round()
    predictions.append(x_test)
    actuals.append(actual)
predictions, actuals = np.vstack(predictions), np.vstack(actuals)    


# In[38]:


rounded_labels=np.argmax(predictions, axis=1)
rounded_labels


# In[39]:


from sklearn.metrics import classification_report

print(classification_report(actuals , rounded_labels, target_names = ['Unidentified Chirality', 'Right Chirality', 'Left Chirality'] ))

