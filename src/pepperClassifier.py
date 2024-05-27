import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time 
import os
import copy
from figurePlot import  perfromPlot, displayData
torch.manual_seed(0)


#Hyperparameters
num_epochs = 25

#Hyperparameters
batch_size = 4
batch_size = 4
learning_rate=0.001 
momentum=0.9
step_size=7
gamma=0.1

# Define data transformations for data augmentation and normalization
std = np.array([0.25, 0.25, 0.25])
mean = np.array([0.5, 0.5, 0.5])

data_transforms = {
    
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        
    ]),
    
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)                          
                               
    ]),   
}
data_dir = '../datasets'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                                for x in ['train', 'val']}

dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                             num_workers=0) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
inputs_Data, classes = next(iter(dataloader['train'])) # obtain a batch of training data

out = torchvision.utils.make_grid(inputs_Data) # Make a grid from batch

displayData(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    start_time = time.time()
    loss_list = []
    accuracy_list = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('*' * 20)
        
        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
                
            runing_loss = 0.0
            runing_corrects = 0
            
            #Iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #Forwad pass
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Perform backward propagation and optimization if its in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                runing_loss +=loss.item() * inputs.size(0)
                runing_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_acc.cpu())
                
        print()
        
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,  loss_list, accuracy_list

#*********Finetuning the network **********
# Load a pretrained model and reset final fully connected layer

model_conv = torchvision.models.resnet18(pretrained = True) # to freeze all the network except the final layer and set requires_grad == False
                                                            # to freeze the parameters in order to prevent gradients computed in backward()
                                                            
for param in model_conv.parameters():
    param.requires_grad = False
    
# The newly constructed modules parameters is set to requires_grad by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum) # only final layer parameters is optimised here contrary to the previious model

#Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

model_conv,  loss_list, accuracy_list = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs) 

# Save the model to model.pt
torch.save(model_conv.state_dict(), '../models/model_conv.pt')

#Plot train cost and validation accuracy,  you can improve results by getting more data.
perfromPlot(loss_list,accuracy_list)

