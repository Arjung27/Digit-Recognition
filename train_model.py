from learning_module import train, test, adjust_learning_rate, to_log_file, get_model
from learning_module import get_transform, plot_fig
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision
import argparse
import numpy as np
import os
import glob
import cv2

def main():
    print("\n_________________________________________________\n")

    parser = argparse.ArgumentParser(description='Digit Recognition')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', nargs='+', default=[100, 150], type=int, help='how often to decrease lr')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--model', default='ResNet18', type=str, help='model for training')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset')
    parser.add_argument('--val_period', default=10, type=int, help='print every __ epoch')
    parser.add_argument('--output', default='output_default', type=str, help='output subdirectory')
    parser.add_argument('--checkpoint', default='check_default', type=str, help='where to save the network')
    parser.add_argument('--model_path', default='', type=str, help='where is the model saved?')
    parser.add_argument('--seed', default=0, type=int, help='seed for seeding random processes.')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--train_augment', dest='train_augment', action='store_true')
    parser.add_argument('--test_augment', dest='test_augment', action='store_true')
    parser.add_argument('--transfer_learning', dest='transfer_learning', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_log = "train_log_{}.txt".format(args.model)
    to_log_file(args, args.output, train_log)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################################################
    #               Dataset

    if args.dataset.lower() == 'mnist':
        train_data = torchvision.datasets.MNIST('./MNIST_dataset/', train=True, download=True,
                                                transform=get_transform(args.train_augment, dataset='MNIST'))
        trainloader = data.DataLoader(train_data, batch_size=args.batch_size)
        test_data = torchvision.datasets.MNIST('./MNIST_dataset/', train=False, download=True,
                                               transform=get_transform(args.test_augment, dataset='MNIST'))
        testloader = data.DataLoader(train_data, batch_size=args.batch_size)
    elif args.dataset.lower() == 'custom':
        train_transform = get_transform(args.train_augment, dataset='CUSTOM')
        test_transform = get_transform(args.test_augment, dataset='CUSTOM')
        train_data = torchvision.datasets.ImageFolder('./training/training',
                                                      transform=train_transform)
        trainloader = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=4)
        test_data = torchvision.datasets.ImageFolder('./validation/validation',
                                                     transform=test_transform)
        testloader = data.DataLoader(train_data, shuffle=False, batch_size=args.batch_size, num_workers=4)

    if args.transfer_learning:
        args.model = 'ResNet18'
        net = models.resnet18(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        net.fc = nn.Linear(512, 10)
    else:
        net = get_model(args.model)

    net = net.to(device)
    print(net)
    start_epoch = 0

    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=2e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if args.model_path != '':
        print("loading model from path: ", args.model_path)
        state_dict = torch.load(args.model_path, map_location=device)
        net.load_state_dict(state_dict['net'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
    ####################################################

    ####################################################
    #        Train and Test
    print("==> Training network...")
    loss = 0
    train_losses = []
    all_acc = []
    test_losses = []
    test_acc_all = []

    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_schedule, args.lr_factor)
        loss, acc = train(net, trainloader, optimizer, criterion, device)

        print(f'Training accuracy at Epoch {epoch}: {acc}')
        print(f'Training loss at Epoch {epoch}: {loss}')
        train_losses.append(loss)
        all_acc.append(acc)

        if (epoch + 1) % args.val_period == 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss)
            print("Training acc: ", acc)
            test_acc, test_loss = test(net, testloader, device, criterion)
            test_losses.append(test_loss)
            test_acc_all.append(test_acc)
            print(" Test accuracy: ", test_acc, "Test Loss: ", test_loss)
            to_log_file({"epoch": epoch, "loss": loss, "training_acc": acc, "test_acc": test_acc},
                        args.output, train_log)

    test_acc, test_loss = test(net, testloader, device, criterion)
    to_log_file({"epoch": epoch, "loss": loss, "test_acc": test_acc}, args.output, train_log)

    #        Save
    state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
            }
    out_str = os.path.join(args.checkpoint, args.model +
                           '_augment_' + str(args.train_augment) +
                           '_optimizer_' + str(args.optimizer) +
                           '_epoch_' + str(epoch) +
                           '_dataset_' + str(args.dataset) +
                           'transfer_learning_' + str(args.transfer_learning) + '.t7')
    print('saving model to: ', args.checkpoint, ' out_str: ', out_str)
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    torch.save(state, out_str)
    ####################################################

    if not os.path.exists(f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}'):
        os.makedirs(f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}')

    indexes_train = np.arange(start_epoch, args.epochs)
    indexes_test = np.arange(args.val_period, args.epochs + 1, args.val_period)
    filename = f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}/training_acc.png'
    plot_fig(indexes_train, all_acc, filename, y_axis='Training Accuracy', x_axis='Epochs')
    filename = f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}/test_acc.png'
    plot_fig(indexes_test, test_acc_all, filename, y_axis='Testing Accuracy', x_axis='Epochs')
    filename = f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}/train_loss.png'
    plot_fig(indexes_train, train_losses, filename, y_axis='Training Loss', x_axis='Epochs')
    filename = f'./plots_dataset={args.dataset}_model={args.model}_transfer_learning={args.transfer_learning}/test_loss.png'
    plot_fig(indexes_test, test_losses, filename, y_axis='Testing Loss', x_axis='Epochs')

    return


if __name__ == "__main__":
    main()