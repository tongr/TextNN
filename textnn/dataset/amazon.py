from pathlib import Path
from typing import Iterable, Tuple

from textnn.utils import read_text_file_lines as read_lines


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
