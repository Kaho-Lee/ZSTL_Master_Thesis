# Zero-Shot Task Transfer Learning Using Relational Attention
## MSc Project on zero-shot task transfer learning using relational attention

ZSTL_GPU.py is the main model file for this project

### Config

- python 3.6.9

- pytorch 1.6.0+cu101

- Jupyter Notebook

- numpy 1.18.5

- pickle 4.0

- sklearn 0.22.2.post1

- tqdm

- matplotlib.pyplot

- sparsemax
Using the implementation from https://github.com/KrisKorrel/sparsemax-pytorch


### Data generation and preprocsessing

1. For AwA and LastFM, 

    1. Run 'dataset_download.sh' to get AwA2 and LastFM dataset

    2. run the relevant feature extraction jupyter notebook at 'Data_Processing' directory to form the tasks dataset

2. For CUB, 
    1. download CUB dataset mannually from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz, place it into ./ZSTL_Data/ and unzip

    2. run CUB_FeatureExtraction.ipynb at 'Data_Processing' directory to get the image feature first, and place the generated file label_feature.txt into ./ZSTL_Data/CUB_200_2011/CUB_200_2011/

    3. run CUB_SingleTaskTrain.ipynb at 'Data_Processing' directory to generate task dataset

### Run experiments

1. For synthetic experiement, run the relevant ZSTL_xxx.ipynb script at the current direcoty

2. For real dataset experiment, place the relevant ZSTL_xxx.ipynb script at 'Experiment' directory into the same directory as this script and run the script