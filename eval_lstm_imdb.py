import logging

from textnn.dataset.imdb import ImdbClassifier

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: {} IMDB_DATA_FOLDER [EMBEDDING_VEC_FILE]".format(sys.argv[0]), file=sys.stderr)
        exit(1)
    logging.basicConfig(level=getattr(logging, "INFO"), format='%(asctime)s %(levelname)s:%(name)s: %(message)s')

    imdb = ImdbClassifier(data_folder=sys.argv[1],
                          pretrained_embeddings_file=sys.argv[2] if len(sys.argv) >= 3 else None,
                          )
    # prepare the model
    imdb.train_or_load_model()

    # debug the text encoder
    imdb.text_enc.print_representations([
        "this is a test is it not?", "this is a test test too", "Unknown word bliblubla",
    ])

    # evaluate the performance of the model
    imdb.evaluate_model()
