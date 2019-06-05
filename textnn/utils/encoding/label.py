from typing import Iterable, List, Union, Tuple

import numpy as np


class LabelEncoder:
    """
    Transforms integer class labels into a binary class matrix.
    """

    def __init__(self, num_classes: int = None, labeled_data: Iterable[Tuple[object, int]] = None,
                 label_sample_it: Iterable[int] = None):
        if not num_classes:
            if label_sample_it is None:
                assert labeled_data is not None, \
                    "Either `num_classes`, `label_sample_it`, or `labeled_data` parameter required!"
                num_classes = max(lab for text, lab in labeled_data) + 1
            else:
                num_classes = max(lab for lab in label_sample_it) + 1

        self.num_classes = num_classes

    @staticmethod
    def integer_class_labels(labeled_data: Iterable[Tuple[object, int]] = None) -> np.ndarray:
        """
        Transforms labled data into integer class labels.
        :param labeled_data: iterable of labeled data (used in case `y_labels` is None)
        :return: an array of class labels
        """
        return np.array(list(lab for text, lab in labeled_data))

    def make_categorical(self, y_labels: Union[np.ndarray, List[int]] = None,
                         labeled_data: Iterable[Tuple[object, int]] = None) -> np.ndarray:
        """
        Transforms integer class labels into a binary class matrix.
        :param y_labels: integer class labels as array
        :param labeled_data: iterable of labeled data (used in case `y_labels` is None)
        :return: a binary class matrix representation of the class label array
        """
        if y_labels is None:
            assert labeled_data, "Either `label_sample_it` or `labeled_data` parameter required!"
            y_labels = list(lab for text, lab in labeled_data)

        import keras.utils
        return keras.utils.to_categorical(y_labels, num_classes=self.num_classes)

    def max_category(self, y_categories: np.ndarray) -> np.ndarray:
        """
        Transforms weighted class matrix into integer class labels (maximum class).
        :param y_categories: a weighted class matrix
        :return: an array of class labels of the maximum class in the weighted class matrix
        """
        assert y_categories.shape[1] == self.num_classes, "Illegal `y_categories` length found (found: {}, " \
                                                          "expected: {})!".format(y_categories.shape[1],
                                                                                  self.num_classes)
        return np.argmax(y_categories, axis=1)
