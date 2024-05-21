#import
import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms, models
import os 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.001
num_classes = 2
batch_size = 4
num_multiprocessing = 4
momentum=0.9
num_epochs = 9

# Define data transformations for data augmentation and normalization

data_transforms = {
    
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]),
    
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                          
                               
    ]),  
    
}

# Define data directory
dataDir = 'datasets'

# create dataset loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), data_transforms[x]) for x in ['train', 'val']}

# Load datasets

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

# Check dataset classes
class_names = image_datasets['train'].classes
print(class_names)


# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

#Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" in name: #unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False
        
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Move the model to the GPU if available

model = model.to(device)


# Training Loop
num_epochs = 9
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs,labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects +=torch.sum(preds == labels.data)
            
            
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        
        print(f'Epoch_{epoch}: {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


print("training is done!")



# Save model
torch.save(model.state_dict(), 'pepper_classification.pth')




                