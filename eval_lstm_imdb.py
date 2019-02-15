import logging

from textnn.dataset.imdb import ImdbClassifier

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, "INFO"), format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    imdb = ImdbClassifier(data_folder="/home/tongr/data/aclImdb/",
                          pretrained_embeddings_file="/home/tongr/data/fastText/wiki.en.align.vec",
                          )
    # prepare the model
    imdb.train_or_load_model()

    # debug the text encoder
    imdb.text_enc.print_representations([
        "this is a test is it not?", "this is a test test too", "Unknown word bliblubla",
    ])

    # evaluate the performance of the model
    imdb.evaluate_model()
