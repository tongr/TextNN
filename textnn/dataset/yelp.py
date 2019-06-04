import logging
import json
from pathlib import Path
from typing import Iterable, Tuple, Union

from textnn.dataset import KerasModelTrainingProgram
from textnn.utils import read_text_file_lines as read_lines, skip


def yelp_star_rating_generator(data_file: Path) -> Iterable[Tuple[str, float]]:
    """
    Generate a text to star-rating tuples from a review.json file.
    :param data_file: the tsv file containing the reviews
    :return: an iterable over the reviews from `data_file`
    """
    def get_text_and_label(line: str) -> Tuple[str, float]:
        data = json.loads(line)
        return data["text"], float(data["stars"])

    return (get_text_and_label(line) for line in read_lines(file_path=data_file))


def yelp_binary_review_generator(data_file: Path, label_3_stars_as=None) -> Iterable[Tuple[str, int]]:
    """
    Generate a text to binary-label tuples from a review.json file.
    :param data_file: the tsv file containing the reviews
    :param label_3_stars_as: specify the binary label for 3-star reviews (if `None`, 3-star reviews are going to be
    ignored)
    :return: an iterable over the reviews from `data_file`
    """
    def stars_to_binary(rating):
        if rating == 3.0:
            return label_3_stars_as
        return 0 if rating < 3.0 else 1

    # transform each rating according to stars_to_binary
    binary_ratings = ((text, stars_to_binary(stars)) for text, stars in yelp_star_rating_generator(data_file=data_file))

    # remove None-labels
    return filter(lambda tup: tup[1] is not None, binary_ratings)


class YelpReviewClassifier(KerasModelTrainingProgram):
    def __init__(self, data_file, vocabulary_size: int = 4096, max_text_length: int = 512,
                 pad_beginning: bool = True, use_start_end_indicators: bool = True,
                 embeddings: Union[int, str, Path] = 16, update_embeddings: bool = True,
                 layer_definitions: str = None,
                 batch_size: int = 1024, num_epochs: int = 100,
                 learning_rate: float = 0.001, learning_decay: float = 0.,
                 shuffle_training_data: Union[int, bool] = 113,
                 log_config: bool = True,
                 ):
        """
        Initialize a new YELP review experiment.
        :param data_file: the file containing the YELP review data (`review.json`)
        :param vocabulary_size: size of the input vocabulary
        :param max_text_length: the maximum amount of token to consider during sequence encoding
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

    def _get_data(self, test_set: bool) -> Iterable[Tuple[str, int]]:
        if not test_set:
            return skip(yelp_binary_review_generator(self._data_file), at_start=self._test_set_skip)

        return skip(yelp_binary_review_generator(self._data_file), at_start=-self._test_set_skip)
