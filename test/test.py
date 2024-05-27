import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os 


# Load saved model

model_conv = torchvision.models.resnet18(pretrained = True) # to freeze all the network except the final layer and set requires_grad == False
                                                            # to freeze the parameters in order to prevent gradients computed in backward()
                                                            
for param in model_conv.parameters():
    param.requires_grad = False
    
# The newly constructed modules parameters is set to requires_grad by default
num_ftrs = model_conv.fc.in_features
print('num_ftrs', num_ftrs)
# Load the model that performs best: 
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("../models/model_conv.pt"))
model.eval()

# Test model with some set of data
#Load and preprocess the unseen image follow the same data transformations used during the training process
image_path = 'test12.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

# perform inferance
with torch.no_grad():
    output = model(input_batch)
    
#Get the predicted class
_, predicted_class = output.max(1)

# Map the predicted class to the class name
class_names = ['bell', 'chilli'] # here is the training class
predicted_class_name = class_names[predicted_class.item()]

print(f'The predicted class is: {predicted_class_name} papper')


# To visualise the predicted image

#Display the image with the predicted class name
image = np.array(image)
plt.imshow(image)
plt.axis('off')
plt.text(10,12, f'Predicted class: {predicted_class_name} pepper', color='white', backgroundcolor='green')
plt.show()