import datetime
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torch.nn as nn
from AlexNet_Custom import *
import torchvision.transforms as transforms
import torchvision
import csv

data_mean = {'mnist' : (0.1307,), 'custom' : ([0.4534, 0.4527, 0.3414])}
data_std = {'mnist' : (0.3081,), 'custom' : ([0.2550, 0.2480, 0.2572])}

def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, 'a') as f:
        f.write(str(out_dict) + "\n")

    print('logging done in ' + out_dir + '.')

def adjust_learning_rate(optimizer, epoch, lr_schedule, lr_factor):
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_factor
        print("Adjusting learning rate ", param_group['lr'] / lr_factor, "->", param_group['lr'])
    return


def test(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            test_outputs = net(inputs)
            loss = criterion(test_outputs, targets)
            test_loss += loss.item()
            _, test_predicted = test_outputs.max(1)
            test_correct += test_predicted.eq(targets).sum().item()
            total += targets.size(0)

    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100. * test_correct / total

    return test_acc, test_loss


def train(net, trainloader, optimizer, criterion, device):

    net.train()
    net = net.to(device)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = train_loss / (batch_idx + 1)
    acc = 100. * correct / total

    return train_loss, acc


def get_transform(augment, dataset="CIFAR10"):
    mean = data_mean[dataset.lower()]
    std = data_std[dataset.lower()]
    cropsize = {'MNIST': 28, 'CUSTOM': 224}
    cropsize = cropsize[dataset]
    padding = None
    if dataset.lower() == 'custom':
        transform_list = [transforms.Resize(224), transforms.CenterCrop(224)]
    else:
        transform_list = []

    if augment:
        transform_list.extend([
            transforms.RandomCrop(cropsize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # print(transform_list)
    return transforms.Compose(transform_list)


def get_model(model, pretrained=False):

    net = eval(model.lower())(pretrained=pretrained)

    return net


def load_model_from_checkpoint(model, model_path, dataset):
    net = get_model(model, dataset)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict['net'])

    except:
        net = torch.nn.DataParallel(net).cuda()
        net.eval()
        print('==> Resuming from checkpoint for %s..' % model)
        checkpoint = torch.load(model_path)
        if 'module' not in list(checkpoint['net'].keys())[0]:
            # to be compatible with DataParallel
            net.module.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint['net'])

    return net


def plot_fig(x_data, y_data, path, y_axis, x_axis):
    fig = plt.figure()
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.plot(x_data, y_data)
    plt.savefig(path)