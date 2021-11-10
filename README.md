# CMU-11685-HW3P2

Fall 2021 Introduction to Deep Learning - Homework 3 Part 2 (RNN-based phoneme recognition)

Author: Ryan Nelson

```text
               _---_
             /       \
            |         |
    _--_    |         |    _--_
   /__  \   \  0   0  /   /  __\
      \  \   \       /   /  /
       \  -__-       -__-  /
   |\   \    __     __    /   /|
   | \___----         ----___/ |
   \                           /
    --___--/    / \    \--___--
          /    /   \    \
    --___-    /     \    -___--
    \_    __-         -__    _/
      ----               ----
       
        O  C  T  O  P  U  S
```

## Introduction

`octopus` is a python module that standardizes the execution of deep learning pipelines using `pytorch`, `wandb`,
and `kaggle`. Module behavior is controlled using a configuration file. The version of `octopus` in this repository has
been updated from the previous version to enable use with LSTMs.

## Requirements

- `wandb` account: Running this code requires a `wandb` account.
- `kaggle.json` file: If you choose to download data from kaggle, you'll need access to an API key file for your kaggle
  account.

## To Run the Code

1. The code for this model consists of the following components:
    - python module `octopus`
    - python module `customized`
    - bash script `mount_drive`

2. Activate an environment that contains torch, torchvision, pandas, numpy, and PIL. I used `pytorch_latest_p37`
   environment as part of the Deep Learning AMI for AWS EC2 instances.

3. Define each configuration file as desired. Configuration file options are listed below. See configuration file
   "config_cnnlstm_r029.txt" as example.

4. If your instance has a mountable storage drive (i.e. `/dev/nvme1n1`), execute the bash script to mount the drive and
   change permissions to allow the user to write to directories inside the drive.

```bash
$ mount_drive 
```

5. Execute the code using the following command from the shell to run `octopus` for each configuration file in the
   configuration directory. Files will be processed in alphabetical order.

```bash
$ python run_octopus.py path/to/config_directory
```

## Running octopus on AWS

1. Launch EC2 instance with `Deep Learning AMI (Ubuntu 18.04) Version 51.0` and `g4dn.2xlarge` instance type.

2. Compress the project directory containing all code.

3. Send the code and `kaggle.json` file to the EC2 instance.

```bash
# copy over code
scp -i /path/to/pem/file.pem /path/to/code/CMU-11685-HW3P2.zip ubuntu@18.217.128.133:/home/ubuntu

# copy over json file
scp -i /path/to/pem/file.pem /path/to/file/kaggle.json ubuntu@18.217.128.133:/home/ubuntu
```

4. Log in, unzip code, and mount drive.

```bash
ssh -i /path/to/pem/file.pem ubuntu@18.217.128.133
unzip CMU-11685-HW3P2.zip
bash CMU-11685-HW3P2/bin/mount_drive
```

5. Use `screen` to multiplex the terminal so training can be moved to the background after starting.

```bash
screen -S any_name_you_want
```

6. Activate desired environment inside this multiplex terminal.

```bash
# activate conda environment
conda activate pytorch_latest_p37
```

7. Execute the wrapper for `octopus`. Note that you'll be asked to log into wandb the first time `octopus` is run on the
   instance.

```bash
python ~/CMU-11685-HW2P2/run_octopus.py /home/ubuntu/CMU-11685-HW2P2/configs/remote
```

8. Move terminal to the background using `CTRL+A,D` and log out of remote server.

## Configuration File Options

Configurations must be parsable by `configparser`.

```text
[DEFAULT]
run_name = CnnLSTM-Run-29  # sets the name of the run in wandb, log, checkpoints, and output files 

[debug]
debug_path = /home/ubuntu/  # where log file will be saved

[kaggle]
download_from_kaggle = True                            # whether to download data from kaggle
kaggle_dir = /home/ubuntu/.kaggle                      # fully qualified location of kaggle directory
content_dir = /home/ubuntu/content/                    # fully qualified location where kaggle data will be downloaded
token_file = /home/ubuntu/kaggle.json                  # where kaggle api file should be placed
competition = idl-fall21-hw2p2s1-face-classification   # the kaggle competition data will be downloaded from
delete_zipfiles_after_unzipping = True                 # whether to delete the zipfiles downloaded from kaggle after unzipping

[pip]
packages=--upgrade wandb==0.10.8,kaggle,Levenshtein,ctcdecode  # commands to install particular pip packages during startup

[wandb]      
wandb_dir = /home/ubuntu/                 # fully qualified directory for wandb internal files
entity = ryanquinnnelson                  # wandb account name
project = CMU-11685-HW2P2A-octopus        # project to save model data under
notes = CNN Face Classification           # any notes to save with this run
tags = CNN,octopus,Resnet34               # any tags to save with this run

[stats]
comparison_metric=val_acc                 # the evaluation metric used to compare this model against previous runs 
comparison_best_is_max=True               # whether a maximum value for the evaluation metric indicates the best performance
comparison_patience=40                    # number of epochs current model can underperform previous runs before early stopping
val_metric_name=val_acc                   # the name of the second metric returned from Evaluation.evalute_model() for clarity in stats

[data]
data_type = numerical                                                                               # indicates data is not image based
data_dir = /home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data                        # fully qualified path to root directory where data subdirectories are located
train_data=/home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data/train.npy              # fully qualified path to training data
train_labels=/home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data/train_labels.npy     # fully qualified path to training labels
val_data=/home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data/dev.npy                  # fully qualified path to validation data
val_labels=/home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data/dev_labels.npy         # fully qualified path to validation labels
test_data=/home/ubuntu/content/competitions/11785-fall2021-hw3p2/HW3P2_Data/test.npy                # fully qualified path to testing data

[output]
output_dir = /home/ubuntu/output         # fully qualified directory where test output should be written

[CTCDecode]
model_path=None      # path to your external kenlm language model(LM). Default is None.
alpha=0              # Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect.
beta=0               # Weight associated with the number of words within our beam.
cutoff_top_n=40      # Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.
cutoff_prob=1.0      # Cutoff probability in pruning. 1.0 means no pruning.
beam_width=1         # This controls how broad the beam search is.
blank_id=0           # This should be the index of the CTC blank token (probably 0).
log_probs_input=True # If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True.

[dataloader]
num_workers=8        # number of workers for use in DataLoader when a GPU is available
pin_memory=True      # whether to use pin memory in DataLoader when a GPU is available
batch_size=256       # batch size regardless of a GPU or CPU

[model]
model_type=CnnLSTM   # type of model to initialize
lstm_input_size=256 # Dimension of features being input into the LSTM portion of the model.
hidden_size=256     # Dimension of each hidden layer in the LSTM model.
num_layers=5        # Number of LSTM layers in LSTM portion of the model.
bidirectional=True  # True if LSTM is bidirectional. False otherwise.
dropout=0.2         # The percent of node dropout in the LSTM model.
lin1_output_size=42 # The number of labels in the feature dimension of the first linear layer if there are multiple linear layers.
lin1_dropout=0.0    # The percent of node dropout in between linear layers in the model if there are multiple linear layers.
output_size=42      # The number of labels in the feature dimension of linear layer output.

# each layer.parameter for the CNN classes or pooling classes 
# 1.padding=1 means the first CNN layer has padding=1
# multiple CNN layers can be listed one after the other (i.e. 1.in_channels=3,..., 2.in_channels=3...)
conv_kwargs=1.in_channels=40,1.out_channels=128,1.kernel_size=3,1.stride=1,1.padding=1,1.dilation=1,2.in_channels=128,2.out_channels=256,2.kernel_size=3,2.stride=1,2.padding=1,2.dilation=1


[checkpoint]
checkpoint_dir = /data/checkpoints   # fully qualified directory where checkpoints will be saved
delete_existing_checkpoints = False  # whether to delete all checkpoints in the checkpoint directory before starting model training (overridden to False if model is being loaded from a previous checkpoint)
load_from_checkpoint=False           # whether model will be loaded from a previous checkpoint
checkpoint_file = None               # fully qualified checkpoint file

[hyperparameters]
num_epochs = 50                                               # number of epochs to train
criterion_type=CTCLoss                                        # the loss function to use
optimizer_type=SGD                                            # optimizer to use
optimizer_kwargs=lr=0.1,weight_decay=0.00005,momentum=0.5     # any optimizer arguments
scheduler_type=ReduceLROnPlateau                              # the scheduler to use
scheduler_kwargs=factor=0.5,patience=3,mode=max,verbose=True  # any scheduler arguments
scheduler_plateau_metric=val_acc                              # if using a plateau-based scheduler, the evaluation metric tracked by the scheduler 
```

## Features

- Allows user to control all module behavior through a single configuration file (see Configuration File Options)
- Can process multiple configuration files in a single execution to facilitate hyperparameter tuning
- (Optionally) downloads and unzips data from kaggle (or is able to use data already available)
- Uses `wandb` to track model statistics, save logs, and calculate early stopping criteria
- Generates a log file for each run
- Automatically saves checkpoints of the model and statistics after each epoch
- Allows user to start running a model from a previously saved checkpoint with minimal effort
- Automatically saves test output after each epoch
- Allows user to customize datasets, evaluation, and test output formatting depending on the data
- Predefined models can be customized via configuration file to facilitate hyperparameter tuning

## To customize the code for a new dataset

Given the understanding that certain aspects of a deep learning model must be customized to the dataset, `octopus`
relies on the user having defined a `customized` python module to manage these aspects. To work smoothly with `octopus`
, `customized` must contain the following files and classes. An example `customized` module is provided with this code.

#### datasets.py

For image data, if training and validation datasets are formatted to facilitate using `ImageFolder`:

- `TestDataset(Dataset)` class implementing the following methods:
    - `__init__(self, test_dir, transform)`
    - `__len__(self) -> int`
    - `__getitem__(self, index) -> Tensor`

For other data:

- `TrainValDataset(Dataset)` class implementing the following methods:
    - `__init__()`
    - `__len__() -> int`
    - `__getitem__() -> Tensor`


- `TestDataset(Dataset)`class implementing the following methods:
    - `__init__()`
    - `__len__() -> int`
    - `__getitem__() -> Tensor`

#### phases.py

- `Training` class implementing the following methods:
    - `__init__(self, val_loader, criterion_func, devicehandler)`
    - `train_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`

- `Evaluation` class implementing the following methods:
    - `__init__(self, val_loader, criterion_func, devicehandler)`
    - `evaluate_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`

- `Testing` class implementing the following methods:
    - `__init__(self, val_loader, criterion_func, devicehandler)`
    - `test_model(self, epoch, num_epochs, model) -> (val_loss, val_metric)`

#### formatters.py

- `OutputFormatter` class implementing the following methods:
    - `__init__(self, data_dir)`
    - `format_output(self, out) -> DataFrame`

## How octopus is used in this project

This project uses LSTMs to predict unaligned phoneme sequences from time-ordered speech frames.

### Data

Each data record represents a single utterance (40 dimension melspectrogram) and its corresponding label is an ordered
list of all phonemes that exist in that utterance. The list of phonemes does not match the number of frames in the
utterance. Utterances have different lengths. Phoneme lists have different lengths as well. We need to pad both the
utterances and labels to obtain a dataset with a single length dimension in order to use pytorch. Additionally, we want
to pack the padded dataset in order to perform efficient calculations inside the LSTM.

The labels use 41 phonemes plus a blank.

### Model

The model consists of an LSTM with CNN layers to extract features, a linear layer to organize the output into a layer of
labels, and a log softmax layer to convert the logits to log probabilities for use in CTCLoss. CTCLoss is used as the
criterion.

### Model Output

Output from the model is a 42-dimension linear layer where each dimension represents one of 42 phonemes. The values of
the linear layer are log probabilities of the different phonemes for a single utterance speech frame. We take all of
these probabilities for a given utterance and perform a "beam search" in order to predict which phonemes exist within
the overall utterance (in order). The result is an array of integers of varying length, where each integer maps to a
single phoneme in our list. Phonemes are listed in order of predicted appearance.

We take the top result of this search and convert the integers into the phoneme encodings, then concatenate the result
into a single string. This predicted string is compared with a target string representing the true phoenemes found in
the utterance. We use Levenshtein distance to calculate the difference.

### Results

The best result I was able to achieve was a Levenshtein distance of 9.97916 using the configuration file "
config_cnnlstm_r029.txt".