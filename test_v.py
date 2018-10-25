from torch.autograd import Variable
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import torchvision
import copy

def test_model(model, dataloaders, dataset_sizes,  use_gpu):
    print("===>Test begains...")
    since = time.time()
    phase='test'

    running_corrects = 0.0
    
    # Iterate over data.
    for data in tqdm(dataloaders[phase]):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs.data, 1)

        #print('Prediction:', preds)
        #print('Labels    :', labels)
        running_corrects += preds.eq(labels).sum().item() 


    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    acc = 100.* running_corrects / dataset_sizes[phase]
    print('Test Acc: {:.4f}'.format(acc))


parser = argparse.ArgumentParser(description='PyTorch inception Training')
parser.add_argument('--img_dir', default='./', type=str, help='')
parser.add_argument('--batch_size', default=1, type=int)

args = parser.parse_args()
data_dir = args.img_dir
batch_size=args.batch_size

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
model_conv=torch.load('./checkpoint/ckpt.t7')

input_shape = 299
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale = 299
input_shape = 299 
use_parallel = True
use_gpu = True

data_transforms = {
        'test': transforms.Compose([
        transforms.Resize((scale,scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}
image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'test'),data_transforms['test'])}
dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,shuffle=False, num_workers=2) }
dataset_sizes = {'test': len(image_datasets['test'])}
class_names = image_datasets['test'].classes

    
print('==> Testing phase')
print('==> Data processing')

test_model(model_conv, dataloaders, dataset_sizes,  use_gpu)
