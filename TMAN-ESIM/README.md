# TMAN-ESIM

## Dependencies
To run it perfectly, you will need (recommend using Ananconda to set up environment):
* Python 3.5 or 3.6
* Tensorflow 1.12.0

1. Download and preprocess


# Download the data and monolingual word embedidng
[1] https://github.com/facebookresearch/XNLI   // Download dataset

[2] https://fasttext.cc/docs/en/pretrained-vectors.html     // Download FastText word embedding

2. Align word embedding by MUSE

# See the README.md in ./TMAN-ESIM/MUSE/ for details

3. Generate word vectors for training

# Merge the aligned word embedding
[1] cat vec.en.txt vec.cn.txt -> vec.en-cn.txt

# Create dev set and test set
[2] python ./src/create_dataset.py

# Create training set
[3] python merge_data.py

# Create word vectors
[4] python ./src/create_word_vectors/create_word_embedding.py --dataset_dir directory to dataset --wordvec_path filepath to word embedding


#Training process

Hyper-parameters are set in configure file in ./config/xnli.sample.config

cd src

python Main.py --config_path ../configs/xnli.sample.config

The model and results are saved in $model_dir$.

#Evaluation

cd src

python Evaluation.py --model_prefix your_model --in_path The path to the test file.

