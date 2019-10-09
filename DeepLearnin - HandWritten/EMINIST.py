import torch
import numpy as np
import matplotlib as plt
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import models

from torch.utils.data.sampler import SubsetRandomSampler

import os
from os.path import exists
from PIL import Image

##### Samuel Amico Fidelis 
##### Última Edição: 24/07/19 ~ 08:36




######### ARQUITETURA DA REDE UTILIZANDO OS PARAMAETROS DA DENSENET121:

model = models.densenet121(pretrained = True)

# Caso tenha GPU:
train_on_gpu = torch.cuda.is_available()

# manter os features parameters
for param in model.parameters():
    param.requires_grad = True


def build_classifier(num_in_features, hidden_layers, num_out_features):
   
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, num_out_features))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(1))
        
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
        classifier.add_module('last', nn.Linear(hidden_layers[-1], num_out_features))
        classifier.add_module('output', nn.LogSoftmax(dim=1))
    return classifier

classifier = build_classifier(1024,[560,410,300,120],2)
#print(classifier)
classifier2 = build_classifier(1024,[1000,950],900)

# move tensors to GPU if CUDA is available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model.fc ---> MUDA O FULLY CONNECTER LAYER, POREM PARA MUDAR ISSO TEM QUE SE ADAPTAR AO CLASSIFIER DA DENSNET121
# A DENSNET 121 SE COMPORTA COM 1024 FEATURES COM SAIDA DE 1000, ACONSELHO MUDAR , POIS É MUITOOOO FRACO
# (classifier): Linear(in_features=1024, out_features=1000, bias=True)
model.classifier = classifier # ESSE MUDA O CLASSIFIER
#model.fc = classifier # ESSE MUDA O FULLY CONECTED LAYER
#print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

if train_on_gpu:
    model.cuda()

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------>   Classificador 1:
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


######## 

# For the bachsize we need to resize all the images in a standard size:





# Lendo o arquivo no PATH:
data_dir = 'Classificador_resize'

"""
Root: 
        |---> Classificador_resize
                |
                |----> train
                |     |---------> Classe1
                |-----> valid
                |     |---------> Classe1
"""


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'valid']}


# Imagens Lidas de uma vez para o vetor de 4D
batch_size = 12
workers = 0

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
              for x in ['train', 'valid']}



# classes da cascata-1
class_names = image_datasets['train'].classes
print("class_names > %s" % (class_names))
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
print(dataset_sizes)



################################################################
############## TREINANDO E VALIDANDO  ##########################
################################################################



### MUDAR OS EPOCHS PARA MAIS CASO QUEIRA MELHORAR
epochs = 20
RL_vector = []
VL_vector = []


for e in range(epochs):
    # keep track of training and validation loss
    running_loss = 0.0
    running_corrects = 0.0
    
    for inputs, label in (dataloaders['train']):
        model.train()
        # IF GPU is availible
        if train_on_gpu:
            inputs, label = inputs.cuda(), label.cuda()
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            logps = model(inputs)
            _, preds = torch.max(logps, 1) # tecnica nova de validacao
            loss = criterion(logps, label)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #print("running_loss = %f  " % (running_loss) )
            running_corrects += torch.sum(preds == label.data)

        
    #else:
    valid_loss = 0.0
    accuracy =0.0
    with torch.no_grad():
        model.eval()
        for inputs, label in (dataloaders['valid']):
            if train_on_gpu:
                inputs, label = inputs.cuda(), label.cuda()
            logps = model(inputs)
            _, preds = torch.max(logps, 1) # tecnica nova de validacao
            loss = criterion(logps, label)
            #print(loss.item()*input.size(0))
            # update average validation loss 
            valid_loss += loss.item()
            #top_p, top_class = ps.topk(1,dim=1)
            #equals = top_class == label.view(*top_class.shape)
            #accuracy += torch.mean(equals.type(torch.FloatTensor)) 
    
        model.train()
        # calculate average losses
        #print("Training LOSS: "  ,running_loss/dataset_sizes['train'] )
        #print("Valid LOSS: " , valid_loss/dataset_sizes['valid'] )
        #print("Accuracy: ", accuracy/dataset_sizes['valid'])
    

    epoch_loss_train = running_loss / dataset_sizes['train']
    epoch_acc_train = running_corrects.double() / dataset_sizes['train']

    epoch_loss_valid = valid_loss / dataset_sizes['valid']
    
    
    #print('{}  \tAcc: {:.4f} \tAcc2: {:.4f}'.format('train', epoch_acc_train,accuracy))


    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAcc:{:.6f}'.format(e, epoch_loss_train, epoch_loss_valid, epoch_acc_train))



################################################################
#########################  TESTANDO  ##########################
################################################################



# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()
i=1
# iterate over test data

for inputs, target in dataloaders['valid']:
    i=i+1
    if len(target)!=batch_size:
        continue
        
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        inputs, target = inputs.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    with torch.no_grad():
        output = model(inputs)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        print(correct_tensor)
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class

    
    for i in range(batch_size):       
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/dataset_sizes['valid']
print('Test Loss: {:.6f}\n'.format(test_loss))

classes = class_names
print("classes > %s" % (classes))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))



### SALVAR MODELO:
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'input_size': 1024,
              'output_size': 2,
              'transform_valid': data_transforms['valid'],
              'epochs': epochs,
              'batch_size': 1,
              'model': models.densenet121(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
   
torch.save(checkpoint, str(classe[0])+'.pth')