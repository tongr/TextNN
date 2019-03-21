import gc
import logging
from pathlib import PurePath, Path
from typing import List, Tuple, Union, Iterable

import numpy as np

from textnn.dataset import KerasModelTrainingProgram
from textnn.utils import read_text_file
from textnn.utils.encoding.text import TokenSequenceEncoder


#
# IMDb specific functions
#
def imdb_file_ref_generator(base_folder, pos_only: bool = None, train_only: bool = None) -> Iterable[Tuple[Path, int]]:
    from pathlib import Path
    from itertools import chain
    base_folder = Path(base_folder)
    assert base_folder.exists(), "Base folder {} does not exist!".format(base_folder)
    for sub_path in ["train/neg", "train/pos", "test/neg", "test/pos"]:
        assert (base_folder / sub_path).exists(), "Data folder {} does not exist!".format(base_folder)

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


class ImdbClassifier(KerasModelTrainingProgram):
    def __init__(self, data_folder, vocabulary_size: int = 4096, max_text_length: int = 512,
                 pad_beginning: bool = True, use_start_end_indicators: bool = True,
                 embeddings: Union[int, str, PurePath] = 32, update_embeddings: bool = True,
                 layer_definitions: str = None,
                 batch_size: int = 1024, num_epochs: int = 25, learning_rate: float = 0.001, learning_decay: float = 0.,
                 shuffle_training_data: Union[int, bool] = 113,
                 log_config: bool = True,
                 ):
        """
        Initialize a new IMDb experiment.
        :param data_folder: the folder containing the IMDb dataset
        :param vocabulary_size: size of the input vocabulary
        :param max_text_length: the maximum amount of token to konsider during sequence encoding
        :param pad_beginning: if True, add padding at start and end of an encoded token sequence
        :param use_start_end_indicators: if True, use reserved indicator token `<START>` and `<END>` during token
        sequence encoding
        :param embeddings: either the size of the embedding layer or the path to a vector file containing pretrained
        embeddings
        :param update_embeddings: if False, the embedding layer will not be updated during training (i.e., for
        pretrained embeddings)
        :param layer_definitions: additional layer definitions downstream the embeddings
        :param batch_size: Number of samples per gradient update
        :param learning_rate: Learning rate
        :param learning_decay: Learning decay
        :param num_epochs: Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y`
        data provided.
        :param shuffle_training_data: shuffle the training to avoid problems in input order (e.g., if data is sorted by
        label). If `shuffle_data=False`, the data will not be shuffled, if `shuffle_data=True`, the data will be
        shuffled randomly, if `shuffle_data` is an integer, this value will be used as seed for the random function.
        :param log_config: if True a the config of this instance is printed after setup
        """
        super().__init__(
            base_folder=data_folder,
            vocabulary_size=vocabulary_size, max_text_length=max_text_length,
            pad_beginning=pad_beginning, use_start_end_indicators=use_start_end_indicators,
            embeddings=embeddings, update_embeddings=update_embeddings,
            layer_definitions=layer_definitions,
            batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate, learning_decay=learning_decay, shuffle_training_data=shuffle_training_data,
        )

        if log_config:
            logging.info(f"{self.__class__.__name__}-configuration:\n{self.config}")

    #
    # public methods
    #
    def train_and_test(self, validation_split: float = .05):
        """
        Train a model (using epoch validation based on `validation_split`) and test it's performance on the independent
        data test set.
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The
        model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and
        any model metrics on this data at the end of each epoch. The validation data is selected from the last samples
        in the `x` and `y` data provided, before shuffling.
        """
        if validation_split:
            self._experiment_folder /= f"validation-split-{validation_split}"
        #
        # training
        #
        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._base_folder, train_only=True))

        # prepare the encoders
        self._prepare_or_load_encoders(
            training_data=training_data,
            initialized_text_enc=TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length,
                pad_beginning=self._pad_beginning,
                add_start_end_indicators=self._use_start_end_indicators,
            ),
        )

        # extract data vectors (from training data)
        text_list = list(tex for tex, lab in training_data)
        x_train: np.ndarray = self._text_enc.encode(texts=text_list)

        # prepare training labels
        y_train: np.ndarray = self._label_enc.make_categorical(labeled_data=training_data)

        # cleanup
        del training_data, text_list

        # load or train model
        self._train_or_load_model(x_train, y_train)

        # cleanup memory
        del x_train, y_train
        gc.collect()

        #
        # testing / evaluate the performance of the model based on the test set
        #
        # extract test data
        def test_data():
            return imdb_data_generator(base_folder=self._base_folder, train_only=False)

        # extract data vectors (from test data)
        x_test: np.ndarray = self._text_enc.encode(texts=list(text for text, lab in test_data()))

        # extract label vectors (from test data)
        y_test_categories: np.ndarray = self._label_enc.make_categorical(labeled_data=test_data())
        gc.collect()

        self._validate_model(x=x_test, y=y_test_categories, validation_file_name="text.json")

        gc.collect()

    def test_encoding(self, *texts: str):
        if len(texts) <= 0:
            logging.warning("Please specify at least one text to encode!")
            return

        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._base_folder, train_only=True))

        # prepare the encoders
        self._prepare_or_load_encoders(
            training_data=training_data,
            initialized_text_enc=TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length,
                pad_beginning=self._pad_beginning,
                add_start_end_indicators=self._use_start_end_indicators,
            ),
        )

        # debug the text encoder
        logging.info(f"Trying to encode the following texts: {texts}")
        self._text_enc.print_representations(texts)

    def cross_validation(self, k: int = 10):
        self._experiment_folder /= f"{k}-fold-cross-validation"
        # get training data
        data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._base_folder, train_only=True))

        # prepare the encoders
        self._prepare_or_load_encoders(
            training_data=data,
            initialized_text_enc=TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length),
        )

        # extract data vectors (from training data)
        text_list = list(tex for tex, lab in data)
        x: np.ndarray = self._text_enc.encode(texts=text_list)

        # prepare training labels
        y_class_labels: np.ndarray = self._label_enc.integer_class_labels(labeled_data=data)

        # cleanup memory
        del text_list, data
        gc.collect()
        self._cross_validation(x=x, y_class_labels=y_class_labels, k=k)
