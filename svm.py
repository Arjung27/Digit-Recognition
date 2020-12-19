import argparse
import os
import sys
import numpy as np
import torch
import torchvision
import  torchvision.datasets
import torch.utils.data as data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def to_log_file(out_dict, out_dir, log_name="results.txt"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as f:
        f.write(str(str(out_dict) + "\n"))

def get_transform(augment=False):

    transform_list = [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,))]

    return torchvision.transforms.Compose(transform_list)

def get_data():

    train_data = torchvision.datasets.MNIST('./MNIST_dataset/', train=True, download=True,
                                            transform=get_transform())
    train_data_loader = data.DataLoader(train_data, batch_size=len(train_data))
    test_data = torchvision.datasets.MNIST('./MNIST_dataset/', train=False, download=True,
                                            transform=get_transform())
    test_data_loader = data.DataLoader(train_data, batch_size=len(test_data))

    train_images = next(iter(train_data_loader))[0].numpy().reshape((len(train_data), -1))
    test_images = next(iter(test_data_loader))[0].numpy().reshape((len(test_data), -1))
    train_labels = next(iter(train_data_loader))[1].numpy().reshape((len(train_data),))
    test_labels = next(iter(test_data_loader))[1].numpy().reshape((len(test_data),))

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SVM Classifier")
    parser.add_argument("--output", default='results/', type=str, help="output directory")
    parser.add_argument('--preprocessing', default=None, type=str, help='Preprocessing for SVM')
    parser.add_argument('--kernel', default=None, type=str, help='rbf, poly, linear')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    result_log = 'results.txt'

    train_images, train_labels, test_images, test_labels = get_data()

    if args.preprocessing is None:
        preprocessing = ['LDA', 'PCA']
    else:
        preprocessing = [args.preprocessing]

    if args.kernel is None:
        kernel_list = ['linear', 'poly', 'rbf']
    else:
        kernel_list = [args.kernel.lower()]

    for transform in preprocessing:
        if transform.upper() == 'PCA':
            func = eval(transform)(n_components=100, svd_solver='randomized', whiten=True)
            train_transformed = func.fit_transform(train_images)
            test_transformed = func.transform(test_images)
        elif transform.upper() == 'LDA':
            func = eval(transform)(n_components=9)
            train_transformed = func.fit_transform(train_images, train_labels)
            test_transformed = func.transform(test_images)

        for kernel in kernel_list:
            print(f"Training SVM classifier using {kernel} kernel and {transform} preprocessing")
            svm_classifier = SVC(kernel=kernel, cache_size=2000)
            svm_classifier.fit(train_transformed, train_labels)
            train_predictions = svm_classifier.predict(train_transformed)
            test_predictions = svm_classifier.predict(test_transformed)
            train_acc = accuracy_score(train_labels, train_predictions)
            test_acc = accuracy_score(test_labels, test_predictions)

            print(f"Training Accuracy: {train_acc}")
            print(f"Test Accuracy: {test_acc}")

            to_log_file(
                {
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "Kernel": kernel,
                    "Preprocessing": transform,
                },
                args.output,
                result_log,
            )
