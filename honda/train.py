#!/usr/bin/env python3


import os
import time
import numpy as np
import torch
from model.efficientnet import EfficientNet
from dataloader.HDD import BuildDataLoader

from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

# fancy progress bar
from tqdm import tqdm, trange

# tensorboardX
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


model_name = 'efficientnet-b2'
BATCH_SIZE = 16
SEED = 44
EPOCHS = 20
lr_rate = 0.001

# torch device :: GPU or CPU training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# data loader
loader = BuildDataLoader()
train_loader, test_loader = loader.build(BATCH_SIZE)

# model building
model = EfficientNet.from_name(model_name)

image_size = EfficientNet.get_image_size(model_name)

# adjust the final linear layer in and out features.
feature = model._fc.in_features
model._fc = torch.nn.Linear(feature,1)

model.to(device)
# summary(model, (1, img_size, img_size))


# ------------------------------------------

torch.manual_seed(SEED)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

# train the model
total_step = len(train_loader)
model.train()
start_time = time.time()
t = trange(EPOCHS)
for epoch in t:
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        batch_loss = 0.0
        images = images.float().to(device)
        labels = torch.reshape(labels, (-1,1)).float().to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)
        batch_loss += loss
        t.set_description("B_Loss : %.5f |" % (loss))
        writer.add_scalar("Loss/train", loss, epoch)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

    epoch_loss += batch_loss

# end of epoch
total_loss.extend(epoch_loss)
t.set_description("Loss : %.5f |" % (epoch_loss))

train_time = time.time() - start_time
print('\n'*3)
print('*'*25)
print(f'\n Time taken to train the model :: {train_time/60:.2f} minutes. \n')
print('*'*25)

writer.flush()

# test the model
model.eval()
with torch.no_grad():
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.float().to(device)
        labels = torch.reshape(labels, (-1,1)).float().to(device)

        output = model(images)
        test_loss = criterion(output, labels)
        total += test_loss  


    print('*'*25)
    print('Test Accuracy of the model on test images: {} %'.format(total))
    print('*'*25)


PATH = './efficientnet/saved_models/'
MODEL = 'honda_eff_b2.pt'
if not os.path.isdir(PATH):
    os.mkdir(PATH) 

torch.save(model, PATH+MODEL)
