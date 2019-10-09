import torch
import numpy as np

import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

import glob
import os
from PIL import Image


def load_checkpoint(filename):
    if os.path.isfile(filename):
        device = torch.device('cpu')
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=device)
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        for parameter in model.parameters():
            parameter.requires_grad = False
    
        print("=> loaded checkpoint '{}' "
                  .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, checkpoint['class_to_idx']

classe = ('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')

def prediction(path,model,topk=2,classe = ('Armas','False')):
    #img = Image.open(path)
    img = path
    transform = transforms.Compose([transforms.Resize(100),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    try:
        
        img = transform(img)
        img = np.expand_dims(img,0)
        img = torch.from_numpy(img)
        model.eval()
        #inputs = Variable(img).to(device)
        logits = model(img)
        _, pred = logits.max(dim=1)
        print('Image predict as:',classe[pred])
        ps = F.softmax(logits,dim=1)
        topk = ps.cpu().topk(topk)
    except:
        return(0,0)

    return (torch.exp(_), pred)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, class_to_idx = load_checkpoint('EMINIST.pth')

'''
for i in range(len(image_process)):
    probs, classes = prediction(image_process[i],model)
'''