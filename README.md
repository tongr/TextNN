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
The recommended EC2 setup (`g2.2xlarge` or better) is based on `Deep Learning AMI (Ubuntu) Version 21.2`
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

#### IMDb - Large Movie Review Dataset
The ACL IMDb dataset consists of 25,000 highly polar movie reviews for training, and 25,000 for testing and can be found 
[here](http://ai.stanford.edu/~amaas/data/sentiment/) ([alt. here](https://www.kaggle.com/pankrzysiu/keras-imdb)).

To run training and evaluation of a LSTM model to predict positive/negative reviews run:
```bash
python ./eval_lstm_imdb.py --data-folder [IMDB_DATA_FOLDER] train-and-evaluate
```
where `[IMDB_DATA_FOLDER]` refers to the base folder of the ACL IMDb dataset. Further optional arguments will influence
the following areas:
 - embedding setup `--pretrained-embeddings-file [PRETRAINED_EMBEDDINGS_FILE]` (`--embed-reserved [True|False]` and/or
   `--retrain-embedding-matrix [True|False]`) or alternatively `--embedding-size [EMBEDDING_SIZE]`
 - text encoding settings `--vocabulary-size [VOCABULARY_SIZE]` and `--max-text-length [MAX_TEXT_LENGTH]`
 - network structure `--lstm-layer-size [LSTM_LAYER_SIZE]`
 - training `--batch-size [BATCH_SIZE]`, `--num-epochs [NUM_EPOCHS]`, `--learning-rate [LEARNING_RATE]`,
   `--learning-decay [LEARNING_DECAY]`, `--shuffle-training-data [True|False|RANDOM_SEED]` (`RANDOM_SEED` refers to an
   `int` value used as the seed for the random number generator)
 - print config information `--log-config [True|False]` (default: `True`)

To debug the selected encoding model run:
```bash
python ./eval_lstm_imdb.py --data-folder [IMDB_DATA_FOLDER] test-encoding "This is a test sentence" \
    "This sentence contains the unknown word klncusuvhacccuuandjccbeddusskxhduscj"
```
Aforementioned optional arguments still apply.

### Pretrained word embeddings
Pretrained word embeddings can be used by loading provided vec files. For instance,
[fastText - aligned word vectors](https://fasttext.cc/docs/en/aligned-vectors.html#vectors) (aalternatively, other
[word vectors](https://fasttext.cc/docs/en/english-vectors.html))

