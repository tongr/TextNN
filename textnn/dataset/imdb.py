import gc
import hashlib
import logging
from pathlib import PurePath, Path
from typing import List, Tuple, Union, Iterable

import numpy as np
from keras import Model
from keras.models import save_model, load_model

from textnn.lstm import train_lstm_classifier
from textnn.utils.encoding import prepare_encoders, LabelEncoder, AbstractTokenEncoder
from textnn.utils.encoding.text import TokenSequenceEncoder, VectorFileEmbeddingMatcher


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


def imdb_data_generator(base_folder, pos_only: bool = None, train_only: bool = None) -> Iterable[Tuple[str, int]]:
    return ((read_text_file(file), lab) for file, lab in imdb_file_ref_generator(base_folder=base_folder,
                                                                                 pos_only=pos_only,
                                                                                 train_only=train_only))


class ImdbClassifier:
    def __init__(self, data_folder, vocabulary_size: int = 5000, max_text_length: int = 1000,
                 embedding_size: int = 32, pretrained_embeddings_file=None, embed_reserved: bool = True,
                 lstm_layer_size: int = 100,
                 batch_size: int = 64, num_epochs: int = 5, shuffle_training_data: Union[int, bool] = 113,
                 ):
        self.data_folder: Path = data_folder if isinstance(data_folder, PurePath) else Path(data_folder)
        self.vocabulary_size = vocabulary_size
        self.max_text_length = max_text_length
        self.embedding_size = embedding_size
        if pretrained_embeddings_file:
            pretrained_embeddings_file = pretrained_embeddings_file if isinstance(pretrained_embeddings_file, PurePath)\
                else Path(pretrained_embeddings_file)
        self.pretrained_embeddings_file: Path = pretrained_embeddings_file
        self.embed_reserved = embed_reserved
        self.lstm_layer_size = lstm_layer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_training_data = shuffle_training_data

        self.model: Model = None
        self.text_enc: AbstractTokenEncoder = None
        self.label_enc: LabelEncoder = None

    @property
    def model_folder(self) -> Path:
        # name sub-folder
        return self.data_folder / ".".join(e for e in [
            # create name by joining all of the following elements with a dot (remove empty strings / None)
            "seq_model",
            f"vocab{self.vocabulary_size}",
            f"pad{self.max_text_length}" if self.max_text_length else None,
            f"emb{self.embedding_size}" if not self.pretrained_embeddings_file else "pretrained_embeddings_{}".format(
                hashlib.md5(open(self.pretrained_embeddings_file, "rb").read()).hexdigest()),
            "embed_reserved" if self.embed_reserved else "",
            f"lstm{self.lstm_layer_size}",
            f"epochs{self.num_epochs}",
            f"batch{self.batch_size}",
            None if self.shuffle_training_data is False else "shuffle({})".format(
                "random" if self.shuffle_training_data is True else self.shuffle_training_data),
            "hd5",
        ] if e)

    @property
    def model_file(self) -> Path:
        return self.model_folder / "keras_model.hd5"

    def train_or_load_model(self):
        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self.data_folder, train_only=True))

        # load encoders and encode training data
        embedding_matcher = None
        if self.pretrained_embeddings_file and self.pretrained_embeddings_file.exists():
            embedding_matcher = VectorFileEmbeddingMatcher(fasttext_vector_file=self.pretrained_embeddings_file,
                                                           encode_reserved_words=self.embed_reserved,
                                                           )

        self.text_enc, self.label_enc, x_train, y_train = prepare_encoders(
            storage_folder=self.model_folder,
            training_data=training_data,
            text_enc_init=lambda: TokenSequenceEncoder(
                limit_vocabulary=self.vocabulary_size,
                default_length=self.max_text_length),
            embedding_matcher=embedding_matcher,
        )

        if self.model_file.exists():
            logging.info(f"Loading models from: {self.model_file}")
            self.model: Model = load_model(str(self.model_file))
        else:
            # train the model
            self.model = train_lstm_classifier(x=x_train, y=y_train,
                                               vocabulary_size=self.text_enc.vocabulary_size,
                                               embedding_size=self.embedding_size,
                                               embedding_matrix=
                                               embedding_matcher.embedding_matrix if embedding_matcher else None,
                                               lstm_layer_size=self.lstm_layer_size,
                                               num_epochs=self.num_epochs, batch_size=self.batch_size,
                                               shuffle_data=self.shuffle_training_data,
                                               )

            gc.collect()
            # serialize data for next time
            save_model(self.model, filepath=str(self.model_file))

        self.model.summary()

        return self.model, self.text_enc, self.label_enc

    def evaluate_model(self):
        #
        # extract test data
        #
        test_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self.data_folder, train_only=False))

        # extract data vectors (from test data)
        x_test: np.ndarray = self.text_enc.encode(texts=list(text for text, lab in test_data))

        # extract label vectors (from test data)
        y_test_categories: np.ndarray = self.label_enc.make_categorical(labeled_data=test_data)
        gc.collect()

        logging.info("Creating predictions ...")
        y_predicted_categories = self.model.predict(x_test, batch_size=self.batch_size)
        gc.collect()

        from sklearn.metrics.classification import accuracy_score, precision_recall_fscore_support
        y_test = self.label_enc.max_category(y_test_categories)
        y_predicted = self.label_enc.max_category(y_predicted_categories)
        logging.info("Results:")
        logging.info("{}".format(precision_recall_fscore_support(y_true=y_test, y_pred=y_predicted)))
        logging.info("{}".format(accuracy_score(y_true=y_test, y_pred=y_predicted)))
