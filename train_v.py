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

        running_corrects += preds.eq(labels).sum().item() 

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    acc = 100.* running_corrects / dataset_sizes[phase]
    print('Test Acc: {:.4f}'.format(acc))

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu, num_epochs, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                #augementation using mixup
                mixup=0
                if phase == 'train' and mixup:
                    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                #running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += preds.eq(labels).sum().item() 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model=copy.deepcopy(model)
                #best_model_wts = model.state_dict()
                
                print("Model Saving...")
                #state = {'net': model.state_dict()}
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                torch.save(model, output_dir+'ckpt_tex.t7')
                print("Model Saved...")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model.state_dict())
    return model



## Loading the dataloaders -- Make sure that the data is saved in following way
"""
data/
  - train/
      - class_1 folder/
          - img1.png
          - img2.png
      - class_2 folder/
      .....
      - class_n folder/
  - val/
      - class_1 folder/
      - class_2 folder/
      ......
      - class_n folder/
"""
parser = argparse.ArgumentParser(description='PyTorch inception Training')
parser.add_argument('--img_dir', default='./', type=str, help='')
parser.add_argument('--output_dir', default='./checkpoint/', type=str, help='')
parser.add_argument('--restore', default=0, type=int, help='whether restore the stored model or not')
parser.add_argument('--final_output', default=2, type=int, help='the number of the output of the final layer')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=20, type=int)

args = parser.parse_args()
data_dir = args.img_dir
restore=args.restore
n_class=args.final_output
batch_size=args.batch_size
epochs=args.epochs
output_dir=args.output_dir

input_shape = 299
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale = 299
input_shape = 299 
use_parallel = True
use_gpu = True

data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((scale,scale)),
        transforms.RandomCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                         shuffle=True, num_workers=2), 
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                         shuffle=False, num_workers=2)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


if(restore):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model_conv=torch.load('./checkpoint/ckpt.t7')
else:
    model_conv = torchvision.models.inception_v3(pretrained=True)
    model_conv.cuda()   

    ## Load the model 
    #model_conv = torchvision.models.inception_v3(pretrained=True)
    ## Lets freeze the first few layers. This is done in two stages 
    # Stage-1 Freezing all the layers 
    freeze_layers=1
    if freeze_layers:
      for i, param in model_conv.named_parameters():
        param.requires_grad = False
    
    # Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_class)
    
    # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    ct = []
    for name, child in model_conv.named_children():
        #if "Conv2d_4a_3x3" in ct:
        for params in child.parameters():
            params.requires_grad = True
        ct.append(name)

    if use_parallel:
        print("[Using all the available GPUs]")
        model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])
    model_conv.cuda()
    
print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss().cuda()

print("[Using small learning rate with momentum...]")
optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)

print("[Creating Learning rate scheduler...]")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("[Training the model begins ....]")
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu,
num_epochs=epochs)

