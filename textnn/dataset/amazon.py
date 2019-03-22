import gc
import logging
import numpy as np

from pathlib import Path
from typing import Iterable, Tuple, Union, List

from textnn.dataset import KerasModelTrainingProgram
from textnn.utils import read_text_file_lines as read_lines, skip
from textnn.utils.encoding.text import TokenSequenceEncoder


def amazon_star_rating_generator(data_file: Path) -> Iterable[Tuple[str, int]]:
    """
    Generate a text to star-rating tuples from a review tsv file.
    :param data_file: the tsv file containing the reviews
    :return: an iterable over the reviews from `dataset_file`
    """
    def get_text_and_label(line: str) -> Tuple[str, int]:
        fields = line.split("\t")
        return " ".join(fields[12:14]), int(fields[7])

    return (get_text_and_label(line) for line in read_lines(file_path=data_file, ignore_first_n_lines=1))


def amazon_binary_review_generator(data_file: Path, label_3_stars_as=None,
                                   ) -> Iterable[Tuple[str, int]]:
    """
    Generate a text to binary-label tuples from a review tsv file.
    :param data_file: the tsv file containing the reviews
    :param label_3_stars_as: specify the binary label for 3-star reviews
    :return: an iterable over the reviews from `dataset_file`
    """
    def stars_to_binary(rating):
        if rating == 3:
            return label_3_stars_as
        return 0 if rating < 3 else 1

    # transform each rating according to stars_to_binary
    binary_ratings = ((text, stars_to_binary(stars)) for text, stars in amazon_star_rating_generator(
        data_file=data_file))

    # remove None-labels
    return filter(lambda tup: tup[1] is not None, binary_ratings)


class AmazonReviewClassifier(KerasModelTrainingProgram):
    def __init__(self, data_file, vocabulary_size: int = 4096, max_text_length: int = 512,
                 pad_beginning: bool = True, use_start_end_indicators: bool = True,
                 embeddings: Union[int, str, Path] = 16, update_embeddings: bool = True,
                 layer_definitions: str = None,
                 batch_size: int = 1024, num_epochs: int = 25, learning_rate: float = 0.001, learning_decay: float = 0.,
                 shuffle_training_data: Union[int, bool] = 113,
                 log_config: bool = True,
                 ):
        """
        Initialize a new Amazon product review experiment.
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
        if not isinstance(data_file, Path):
            data_file = Path(data_file)
        assert data_file.exists(), f"Unable to find specified dataset in '{data_file}'!"
        super().__init__(
            base_folder=data_file.parent,
            vocabulary_size=vocabulary_size, max_text_length=max_text_length,
            pad_beginning=pad_beginning, use_start_end_indicators=use_start_end_indicators,
            embeddings=embeddings, update_embeddings=update_embeddings,
            layer_definitions=layer_definitions,
            batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate, learning_decay=learning_decay, shuffle_training_data=shuffle_training_data,
        )
        self._data_file = data_file
        self._test_set_skip = 1000

        if log_config:
            logging.info(f"{self.__class__.__name__}-configuration:\n{self.config}")

    def _get_data(self, training_set: bool) -> Iterable[Tuple[str, int]]:
        if training_set:
            return skip(amazon_binary_review_generator(self._data_file), at_start=self._test_set_skip)

        return skip(amazon_binary_review_generator(self._data_file), at_start=-self._test_set_skip)

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
        print(f"validation-split {validation_split}")
        if validation_split:
            self._experiment_folder /= f"validation-split-{validation_split}"
        #
        # training
        #
        # get training data
        training_data: List[Tuple[str, int]] = list(self._get_data(training_set=True))

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
        self._train_or_load_model(x_train, y_train, validation_split=validation_split)

        # cleanup memory
        del x_train, y_train
        gc.collect()

        #
        # testing / evaluate the performance of the model based on the test set
        #

        # extract data vectors (from test data)
        x_test: np.ndarray = self._text_enc.encode(
            texts=list(text for text, lab in self._get_data(training_set=False)))

        # extract label vectors (from test data)
        y_test_categories: np.ndarray = self._label_enc.make_categorical(
            labeled_data=self._get_data(training_set=False))
        gc.collect()

        self._validate_model(x=x_test, y=y_test_categories, validation_file_name="text.json")

        gc.collect()

