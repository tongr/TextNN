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
docker run -v "${PWD}:/code:ro" -w "/code" -it textnn
```

## Datasets

#### IMDb - Large Movie Review Dataset
The ACL IMDb dataset consists of 25,000 highly polar movie reviews for training, and 25,000 for testing and can be found 
[here](http://ai.stanford.edu/~amaas/data/sentiment/) ([alt. here](https://www.kaggle.com/pankrzysiu/keras-imdb)).

To ru training and evaluation of a LSTM model to predict positive/negative reviews run:
```bash
python ./eval_lstm_imdb.py
```
