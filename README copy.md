# Get Start
Here is an instruction of how to use this code. 
## Data Preparation
```
cd hw3p2
kaggle competitions download -c 11785-fall2021-hw3p2
unzip 11785-fall2021-hw3p2.zip
rm 11785-fall2021-hw3p2.zip
```
## Required Packages
```
# install ctcdecoder
cd hw3p2
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
# Install other required packages, I maynot list all required ones
# Just install if you get "No Module Named xxx"
pip install argparse
pip install tqdm
pip install Levenshtein
pip install numpy
pip install pandas
```
## Train:
You can have your own configs and pass it train.py through the --args value.

I just make basic settings in the argparser, you should add more
```
python train.py --args value
```
## Inference:
```
python test.py --args value
```
