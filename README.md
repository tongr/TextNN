# TextNN
TextNN is a collection of Python code snippets solving different text mining tasks (on varying datasets) using deep learning.

## Installation
Before using the code, please install the necessary software dependencies.
 - Install conda (i.e., [Anaconda](https://docs.anaconda.com/anaconda/install/) or
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
 - Create `textnn` conda environment:
   ```bash
   conda env create -f environment.yml; \
   conda activate textnn
   ```
   Update the conda environment (from an old version):
   ```bash
   conda env update -f environment.yml; \
   conda activate textnn
   ```

### Docker
Running the code in a docker container can be achieved by building the image:
```bash
docker build -t textnn .
```
and running the image in interactive mode (conda environment automatically loaded)
```bash
docker run --rm --runtime=nvidia -v "${PWD}:/code" -w "/code" -it textnn
```

### AWS
The recommended EC2 setup (e.g., `g3s.xlarge`) is based on `Deep Learning AMI (Ubuntu) Version 21.2`
(ami-0e9085a8d461c2d01) with an increased volumne of 120GB or more. It is recommended to execute code via
[Docker](#Docker), by setting up the project and creating an image:
```bash
git clone https://github.com/tongr/TextNN && cd TextNN && \
    docker build -t textnn .
```
And running the experiments inside the container:
```bash
docker run --rm --runtime=nvidia -v "${PWD}:/code" -w "/code" -it textnn
```

## Datasets

### Labeled data
The individual datasets have a specific `DATASET` indicator, the parameters for the following experiments are
equivalent:

 1. To run training and evaluation of a LSTM model to predict positive/negative reviews run:
    ```bash
    python ./run_experiment.py [DATASET] [OPT_ARGS] train-and-test [--validation-split VALIDATION_HOLD_OUT_RATIO]
    ```
    where the optional `VALIDATION_HOLD_OUT_RATIO` (default `0.05`) specified how much data will be hold back for epoch
    validation during training.

    Further optional arguments `OPT_ARGS` will influence the following areas:
     - text encoding settings: `--vocabulary-size VOCABULARY_SIZE`, `--max-text-length MAX_TEXT_LENGTH`,
       `--pad-beginning [True|False]` (whether to add padding at start and end of a sequence), and
       `--use-start-end-indicators [True|False]` (whether to use reserved indicator token `<START>` and `<END>`)
     - embedding setup: `--embeddings [EMBEDDING_SIZE|PRETRAINED_EMBEDDINGS_FILE]` (`--update-embeddings [True|False]`)
     - network structure `--layer_definitions [LAYER_DEFINITIONS]` (layer definitions separated by pipe, e.g.,
       `--layer-definitions 'LSTM(16)|Dense(8)'`)
     - training `--batch-size BATCH_SIZE`, `--num-epochs NUM_EPOCHS`, `--learning-rate LEARNING_RATE`,
       `--learning-decay LEARNING_DECAY`, `--shuffle-training-data [True|False|RANDOM_SEED]` (`RANDOM_SEED` refers to an
       `int` value used as the seed for the random number generator)
     - print config information `--log-config [True|False]` (default: `True`)
 1. To debug the selected encoding model run:
    ```bash
    python ./run_experiment.py [DATASET] [OPT_ARGS] \
        test-encoding "This is a test sentence" "This sentence contains the unknown word klcuvhacnjbduskxuscj"
    ```
    This command will create representations for the two example sentences.
    Aforementioned optional arguments `OPT_ARGS` still apply.
 1. To execute *k*-fold cross validation based only on the training data set
    ```bash
    python ./run_experiment.py [DATASET] [OPT_ARGS] \
        cross-validation [--k NUMBER_OF_FOLDS]
    ```
    The `NUMBER_OF_FOLDS` indicates the amout of folds / splits to use for cross validation.
    Aforementioned optional arguments `OPT_ARGS` still apply.

#### IMDb - Large Movie Review Dataset
The ACL IMDb dataset consists of 25,000 highly polar movie reviews for training, and 25,000 for testing and can be found 
[here](http://ai.stanford.edu/~amaas/data/sentiment/) ([alt. here](https://www.kaggle.com/pankrzysiu/keras-imdb)).

Preparation:
   Download dataset and extract it in the aclImdb subfolder.
   ```bash
   curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar -xz
   ```
   In the following examples, the indicator `IMDB_DATA_FOLDER` refers to the base folder of the ACL IMDb dataset:
   `IMDB_DATA_FOLDER=${PWD}/aclImdb/`

Run experiments:
 1. To run training and evaluation of a LSTM model to predict positive/negative reviews run:
    ```bash
    python ./run_experiment.py imdb --data-folder [IMDB_DATA_FOLDER] [OPT_ARGS] \
       train-and-test [--validation-split VALIDATION_HOLD_OUT_RATIO]
    ```
    where `IMDB_DATA_FOLDER` refers to the base folder of the ACL IMDb dataset and the aforementioned optional
    arguments `VALIDATION_HOLD_OUT_RATIO` and `OPT_ARGS` still apply.

 1. To debug the selected encoding model run:
    ```bash
    python ./run_experiment.py imdb --data-folder [IMDB_DATA_FOLDER] [OPT_ARGS] \
       test-encoding "This is a test sentence" "This sentence contains the unknown word klcuvhacnjbduskxuscj"
    ```
    The aforementioned optional arguments `OPT_ARGS` still apply.

 1. To execute *k*-fold cross validation based only on the training data set
    ```bash
    python ./run_experiment.py imdb --data-folder [IMDB_DATA_FOLDER] [OPT_ARGS] \
        cross-validation [--k NUMBER_OF_FOLDS]
    ```
    The aforementioned optional arguments `NUMBER_OF_FOLDS` and `OPT_ARGS` still apply.

#### Amazon Customer Reviews
The [Amazon reviews dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) consists of hundred million
reviews by millions of Amazon customers over two decades. The reviews express opinions and describe the customer
experiences regarding products on the Amazon.com website. Different review subsets are listed here:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt

Preparation:
    Download a dataset (e.g., Amazon Video reviews `amazon_reviews_us_Video_v1_00.tsv.gz`):
    ```
    wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_v1_00.tsv.gz
    ```
    In the following examples, the indicator `AMAZON_DATA_FILE` refers to the downloaded data file of the Amazon
    dataset:
    `AMAZON_DATA_FILE=${PWD}/amazon_reviews_us_Video_v1_00.tsv.gz`

Run experiments:
 1. To run training and evaluation of a LSTM model to predict positive/negative reviews run:
    ```bash
    python ./run_experiment.py amazon --data-file [AMAZON_DATA_FILE] [OPT_ARGS] \
        train-and-test [--validation-split VALIDATION_HOLD_OUT_RATIO]
    ```
    where `AMAZON_DATA_FILE` refers to the Amazon dataset file, the aforementioned optional arguments
    `VALIDATION_HOLD_OUT_RATIO` and `OPT_ARGS` still apply.

#### YELP reviews
TODO: add description ...

#### DBpedia categories
TODO: add description ...

#### Yahoo! Answers
TODO: add description ...

#### AG news
TODO: add description ...

#### Sogou news 
TODO: add description ...

### Pretrained word embeddings
Pretrained word embeddings can be used by loading provided vec files. For instance,
[fastText - aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html#vectors) (aalternatively, other
[word vectors](https://fasttext.cc/docs/en/english-vectors.html))

