# Digit-Recognition

In order to use/run the code kindly first install the dependencies. The code is tested on:

```
Ubunut 18.04
Python 3.8.2
PyTorch 1.14.0
```

Install all the dependencies by running the following command from the root folder in your shell:

```
pip install -r requirements.txt
```

The code autmotically downloads the MNIST dataset but you must place the dataset for task 2 in the training and the validation folder in the root directory. To run the svm code, run the following command:

```
python svm.py
```

To run the code for task 1 using AlexNet, use the following command for baseline results:

```
python train_model.py --train_augment --model ALEXNET_MNIST --batch_size 64 --epochs 20 --lr_schedule 10 --val_period 4
```

To run the simple CNN architecture for task 2, use the following command for baseline results:

```
python train_model.py --train_augment --model ALEXNET_CUSTOM --batch_size 64 --epochs 40 --lr_schedule 20 --val_period 4 --dataset custom --lr 0.005
```

To run the tranfer learning setup for task 2, use the following command for baseline results:
```
python train_model.py --train_augment --batch_size 64 --epochs 20 --lr_schedule 10 --val_period 2 --transfer_learning --dataset custom --lr 0.01
```
