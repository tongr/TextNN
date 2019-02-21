import gc
import hashlib
import logging
from pathlib import PurePath, Path
from typing import List, Tuple, Union, Iterable, Any

import numpy as np
from keras import Model
from keras.models import save_model, load_model

from textnn.lstm import train_lstm_classifier
from textnn.utils import plot2file, join_name
from textnn.utils.encoding import prepare_encoders, LabelEncoder, AbstractTokenEncoder
from textnn.utils.encoding.text import TokenSequenceEncoder, VectorFileEmbeddingMatcher


#
# IMDb specific functions
#
def read_text_file(file_path: Path) -> str:
    with open(str(file_path), 'r', encoding='utf8') as file:
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
                 log_config: bool = True,
                 ):
        self._data_folder: Path = data_folder if isinstance(data_folder, PurePath) else Path(data_folder)
        self._vocabulary_size = vocabulary_size
        self._max_text_length = max_text_length
        self._embedding_size = embedding_size
        if pretrained_embeddings_file:
            pretrained_embeddings_file = pretrained_embeddings_file if isinstance(pretrained_embeddings_file, PurePath)\
                else Path(pretrained_embeddings_file)
        self._pretrained_embeddings_file: Path = pretrained_embeddings_file
        self._embed_reserved = embed_reserved
        self._lstm_layer_size = lstm_layer_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle_training_data = shuffle_training_data

        self._model: Model = None
        self._text_enc: AbstractTokenEncoder = None
        self._label_enc: LabelEncoder = None
        if log_config:
            logging.info(f"{self.__class__.__name__}-configuration:\n{self.config}")

    @property
    def _config_pairs(self) -> Iterable[Tuple[str, Any]]:
        from itertools import chain
        kv = chain(
            self.__dict__.items(),
            [("_encoder_folder", self._encoder_folder), ("_model_folder", self._model_folder)])
        return ((key.lstrip("_"), value) for key, value in kv)

    @property
    def config(self) -> str:
        return "\n".join(f"  {key}: {value}" for key, value in sorted(self._config_pairs))

    @property
    def _encoder_folder(self) -> Path:
        # name sub-folder
        return self._data_folder / join_name([
            # create name by joining all of the following elements with a dot (remove empty strings / None)
            "sequences",
            f"vocab{self._vocabulary_size}",
            f"pad{self._max_text_length}" if self._max_text_length else None,
        ])

    @property
    def _model_folder(self) -> Path:
        # name sub-folder
        return self._encoder_folder / join_name([
            # create name by joining all of the following elements with a dot (remove empty strings / None)
            "lstm",
            f"emb{self._embedding_size}" if not self._pretrained_embeddings_file else "pretrained_embeddings_{}".format(
                hashlib.md5(open(str(self._pretrained_embeddings_file), "rb").read()).hexdigest()),
            "embed_reserved" if self._embed_reserved else "",
            f"lstm{self._lstm_layer_size}",
            f"epochs{self._num_epochs}",
            f"batch{self._batch_size}",
            None if self._shuffle_training_data is False else "shuffle" if self._shuffle_training_data is True
            else f"shuffle{self._shuffle_training_data}",
        ])

    def _train_or_load_model(self):
        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._data_folder, train_only=True))

        # load encoders and encode training data
        embedding_matcher = None
        if self._pretrained_embeddings_file and self._pretrained_embeddings_file.exists():
            embedding_matcher = VectorFileEmbeddingMatcher(fasttext_vector_file=self._pretrained_embeddings_file,
                                                           encode_reserved_words=self._embed_reserved,
                                                           )
        self._text_enc, self._label_enc, x_train, y_train = prepare_encoders(
            storage_folder=self._encoder_folder,
            training_data=training_data,
            text_enc_init=lambda: TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length),
            embedding_matcher=embedding_matcher,
        )

        model_file = self._model_folder / "keras_model.hd5"
        if model_file.exists():
            logging.info(f"Loading models from: {model_file}")
            self._model: Model = load_model(str(model_file))
        else:
            # this model is inspired by the configuration of Susan Li:
            # https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e

            # train the model
            self._model, history = train_lstm_classifier(
                x=x_train, y=y_train,
                vocabulary_size=self._text_enc.vocabulary_size,
                embedding_size=self._embedding_size,
                embedding_matrix=
                embedding_matcher.embedding_matrix if embedding_matcher else None,
                lstm_layer_size=self._lstm_layer_size,
                num_epochs=self._num_epochs, batch_size=self._batch_size,
                shuffle_data=self._shuffle_training_data,
            )
            # plot accuracy
            plot2file(
                file=self._model_folder / "accuracy.png",
                x_values=list(range(self._num_epochs)),
                y_series={
                    "Training acc": history.history['acc'],
                    "Validation acc": history.history['val_acc'],
                },
                title="Training and validation accuracy",
                x_label="Epochs",
                y_label="Accuracy",
            )
            # plot loss
            plot2file(
                file=self._model_folder / "loss.png",
                x_values=list(range(self._num_epochs)),
                y_series={
                    "Training loss": history.history['loss'],
                    "Validation loss": history.history['val_loss'],
                },
                title="Training and validation loss",
                x_label="Epochs",
                y_label="Loss",
            )

            gc.collect()
            # serialize data for next time
            save_model(self._model, filepath=str(model_file))

        self._model.summary()

        return self._model, self._text_enc, self._label_enc

    def _evaluate_model(self):
        #
        # extract test data
        #
        test_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._data_folder, train_only=False))

        # extract data vectors (from test data)
        x_test: np.ndarray = self._text_enc.encode(texts=list(text for text, lab in test_data))

        # extract label vectors (from test data)
        y_test_categories: np.ndarray = self._label_enc.make_categorical(labeled_data=test_data)
        gc.collect()

        logging.info("Creating predictions ...")
        y_predicted_categories = self._model.predict(x_test, batch_size=self._batch_size)
        gc.collect()

        from sklearn.metrics.classification import accuracy_score, precision_recall_fscore_support
        y_test = self._label_enc.max_category(y_test_categories)
        y_predicted = self._label_enc.max_category(y_predicted_categories)
        logging.info("Results:")
        logging.info("{}".format(precision_recall_fscore_support(y_true=y_test, y_pred=y_predicted)))
        logging.info("{}".format(accuracy_score(y_true=y_test, y_pred=y_predicted)))

    def train_and_evaluate(self):
        # prepare the model
        self._train_or_load_model()

        # evaluate the performance of the model
        self._evaluate_model()

    def test_encoding(self, *texts: Union[Tuple[str], List[str]]):
        if len(texts) <= 0:
            logging.warning("Please specify at least one text to encode!")
            return

        logging.info(f"Trying to encode the following texts: {texts}")
        # prepare the model
        self._train_or_load_model()

        # debug the text encoder
        self._text_enc.print_representations(texts)
