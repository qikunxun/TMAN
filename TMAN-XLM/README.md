# TMAN-XLM


XLM supports multi-GPU and multi-node training.


## Pretrained models

XLM provides pretrained cross-lingual language models, all trained with the MLM objective (see training command below):

| Languages        | Model                                                               | BPE codes                                                     | Vocabulary                                                     |
| ---------------- |:-------------------------------------------------------------------:|:-------------------------------------------------------------:| --------------------------------------------------------------:|
| XNLI-15          | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth) | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) |


## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/glample/fastBPE) (generate and apply BPE codes)
- [Moses](http://www.statmt.org/moses/) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://www.github.com/nvidia/apex) (for fp16 training)


## Cross-lingual text classification (XNLI)

XLMs can be used to build cross-lingual classifiers. After fine-tuning an XLM model on an English training corpus for instance (e.g. of sentiment analysis, natural language inference), the model is still able to make accurate predictions at test time in other languages, for which there is very little or no training data. This approach is usually referred to as "zero-shot cross-lingual classification".

### Get the right tokenizers

Before running the scripts below, make sure you download the tokenizers from the [tools/](https://github.com/facebookresearch/XLM/tree/master/tools) directory.

### Download / preprocess monolingual data

This script will download and preprocess the Wikipedia datasets in the 15 languages that are part of XNLI:

```
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  ./get-data-wiki.sh $lg
done
```

Downloading the Wikipedia dumps make take several hours. The *get-data-wiki.sh* script will automatically download Wikipedia dumps, extract raw sentences, clean and tokenize them, apply BPE codes and binarize the data. Note that in our experiments we also concatenated the [Toronto Book Corpus](http://yknzhu.wixsite.com/mbweb) to the English Wikipedia.

For Chinese and Thai you will need a special tokenizer that you can install using the commands below. For all other languages, the data will be tokenized with Moses scripts.

```
# Thai - https://github.com/PyThaiNLP/pythainlp
pip install pythainlp

# Chinese
cd tools/
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```

### Download / preprocess parallel data

This script will download and preprocess parallel data that can be used for the TLM objective:

```
lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
for lg_pair in $lg_pairs; do
  ./get-data-para.sh $lg_pair
done
```

### Download / preprocess XNLI data

This script will download and preprocess the XNLI corpus:


# Replace multinli.train.en.tsv by the merged data

```
./get-data-xnli.sh
```


### Train on XNLI from a pretrained model

You can now use the pretrained model for cross-lingual classification. To download a model trained with the command above on the MLM-TLM objective, run:

```
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth
```

You can now fine-tune the pretrained model on XNLI, or on one of the English GLUE tasks:

```
python glue-xnli.py
--exp_name test_xnli_mlm_tlm             # experiment name
--dump_path ./dumped/                    # where to store the experiment
--model_path mlm_tlm_xnli15_1024.pth     # model location
--data_path ./data/processed/XLM15       # data location
--transfer_tasks XNLI                    # transfer tasks (XNLI or GLUE tasks)
--optimizer adam,lr=0.000005             # optimizer
--batch_size 8                           # batch size
--n_epochs 2                             # number of epochs
--epoch_size -1                          # number of sentences per epoch
--max_len 128                            # max number of words in sentences
--max_vocab 95000                        # max number of words in vocab
```

