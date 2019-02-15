import gc
import hashlib
import logging
from pathlib import PurePath, Path
from typing import List, Tuple, Any, Union, Iterable

import numpy as np
from keras import Model
from keras.models import save_model, load_model

from textnn.lstm import train_lstm_classifier
from textnn.utils.encoding import prepare_encoders, LabelEncoder, AbstractTokenEncoder
from textnn.utils.encoding.text import TokenSequenceEncoder, VectorFileEmbeddingMatcher, print_representations


#
# IMDb specific functions
#
def read_text_file(file_path: Path) -> str:
    with open(str(file_path), 'r') as file:
        data = file.read()
    return data


def imdb_file_ref_generator(base_folder, pos_only: bool = None, train_only: bool = None) -> Iterable[Tuple[Path, int]]:
    from pathlib import Path
    from itertools import chain
    base_folder = Path(base_folder)
    assert base_folder.exists(), "Base folder {} does not exist!".format(base_folder)
    for subpath in ["train/neg", "train/pos", "test/neg", "test/pos"]:
        assert (base_folder / subpath).exists(), "Data folder {} does not exist!".format(base_folder)

    result_generators = []
    if pos_only is None or pos_only:
        # positive class
        if train_only is None or train_only:
            result_generators.append((file_path, 1) for file_path in (base_folder / "train/pos").glob("*.txt"))

        if train_only is None or not train_only:
            result_generators.append((file_path, 1) for file_path in (base_folder / "test/pos").glob("*.txt"))

    if pos_only is None or not pos_only:
        # negative class
        if train_only is None or train_only:
            result_generators.append((file_path, 0) for file_path in (base_folder / "train/neg").glob("*.txt"))

        if train_only is None or not train_only:
            result_generators.append((file_path, 0) for file_path in (base_folder / "test/neg").glob("*.txt"))

    return chain(*result_generators)


def read_imdb_data(base_folder, pos_only: bool = None, train_only: bool = None) -> List[Tuple[str, int]]:
    return list(imdb_data_generator(base_folder=base_folder, pos_only=pos_only, train_only=train_only))


def imdb_data_generator(base_folder, pos_only: bool = None, train_only: bool = None) -> Iterable[Tuple[str, int]]:
    return ((read_text_file(file), lab) for file, lab in imdb_file_ref_generator(base_folder=base_folder,
                                                                                 pos_only=pos_only,
                                                                                 train_only=train_only))


def train_or_load_imdb_model(data_folder: Any, vocabulary_size=5000, max_text_length: int = None,
                             embedding_size=32,
                             fasttext_embedding_file: Union[str, Path] = None, embed_reserved: bool = False,
                             lstm_layer_size: int = 100,
                             num_epochs: int = 5, batch_size: int = 64, shuffle_training_data: Union[int, bool] = 113,
                             ) -> Tuple[Model, AbstractTokenEncoder, LabelEncoder]:
    if not isinstance(data_folder, PurePath):
        data_folder = Path(data_folder)

    # name sub-folder
    model_folder = data_folder / ".".join(e for e in [
        # create name by joining all of the following elements with a dot (remove empty strings / None)
        "seq_model",
        f"vocab{vocabulary_size}",
        f"pad{max_text_length}" if max_text_length else None,
        f"emb{embedding_size}" if not fasttext_embedding_file else "fastText_{}".format(
            hashlib.md5(open(fasttext_embedding_file, "rb").read()).hexdigest()),
        "embed_reserved" if embed_reserved else "",
        f"lstm{lstm_layer_size}",
        f"epochs{num_epochs}",
        f"batch{batch_size}",
        None if shuffle_training_data is False else "shuffle({})".format(
            "random" if shuffle_training_data is True else shuffle_training_data),
        "hd5",
    ] if e)

    #
    # get training data
    #
    training_data: List[Tuple[str, int]] = read_imdb_data(base_folder=data_folder, train_only=True)

    # load encoders and encode training data
    embedding_matcher = None
    if fasttext_embedding_file and Path(fasttext_embedding_file).exists():
        embedding_matcher = VectorFileEmbeddingMatcher(fasttext_vector_file=fasttext_embedding_file,
                                                       encode_reserved_words=embed_reserved,
                                                       )

    text_enc, label_enc, x_train, y_train = prepare_encoders(model_folder=model_folder,
                                                             training_data=training_data,
                                                             text_enc_init=lambda: TokenSequenceEncoder(
                                                                 limit_vocabulary=vocabulary_size,
                                                                 default_length=max_text_length),
                                                             embedding_matcher=embedding_matcher,
                                                             )

    model_file = model_folder / "keras_model.hd5"

    if model_file.exists():
        logging.info(f"Loading models from: {model_file}")
        model: Model = load_model(str(model_file))
    else:
        # train the model
        model = train_lstm_classifier(x=x_train, y=y_train,
                                      vocabulary_size=text_enc.vocabulary_size,
                                      embedding_size=embedding_size,
                                      embedding_matrix=embedding_matcher.embedding_matrix if embedding_matcher else None,
                                      lstm_layer_size=lstm_layer_size,
                                      num_epochs=num_epochs, batch_size=batch_size, shuffle_data=shuffle_training_data,
                                      )

        gc.collect()
        #
        # serialize data for next time
        #
        save_model(model, filepath=str(model_file))

    return model, text_enc, label_enc


def train_and_test_imdb_model(imdb_folder: str, fasttext_file: str):
    batch_size = 64
    m, text_encoder, label_encoder = train_or_load_imdb_model(data_folder=imdb_folder,
                                                              vocabulary_size=5000, max_text_length=1000,
                                                              fasttext_embedding_file=fasttext_file,
                                                              embed_reserved=True,
                                                              # embedding_size=32,
                                                              lstm_layer_size=100, batch_size=batch_size, num_epochs=3,
                                                              )

    m.summary()

    print_representations(["this is a test is it not?", "this is a test test too", "Unknown word bliblubla"],
                          text_encoder)
    #
    # extract test data
    #
    test_data: List[Tuple[str, int]] = read_imdb_data(base_folder=imdb_folder, train_only=False)

    # extract data vectors (from test data)
    X_test: np.ndarray = text_encoder.encode(texts=list(text for text, lab in test_data))

    # extract label vectors (from test data)
    y_test_categories: np.ndarray = label_encoder.make_categorical(labeled_data=test_data)
    gc.collect()

    logging.info("Creating predictions ...")
    y_predicted_categories = m.predict(X_test, batch_size=batch_size)
    gc.collect()

    from sklearn.metrics.classification import accuracy_score, precision_recall_fscore_support
    y_test = label_encoder.max_category(y_test_categories)
    y_predicted = label_encoder.max_category(y_predicted_categories)
    logging.info("Results:")
    logging.info("{}".format(precision_recall_fscore_support(y_true=y_test, y_pred=y_predicted)))
    logging.info("{}".format(accuracy_score(y_true=y_test, y_pred=y_predicted)))
    # # enc.prepare_vocabulary(["this is a test is it not?", "this is a test test too"], False)
    # print_representations(["this is a test is it not?", "this is a test test too", "Unknown word bliblubla"],
    #                       text_encoder)
