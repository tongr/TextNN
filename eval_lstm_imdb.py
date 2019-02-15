import logging

from textnn.dataset.imdb import train_and_test_imdb_model

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, "INFO"), format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    train_and_test_imdb_model(imdb_folder="/home/tongr/data/aclImdb/",
                              fasttext_file="/home/tongr/data/fastText/wiki.en.align.vec"
                              )
