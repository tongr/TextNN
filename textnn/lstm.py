import logging
from typing import Any, List, Union, Tuple

import numpy as np
from keras import Sequential
from keras.engine.base_layer import Layer
from keras.layers import Embedding, Dense
from keras.optimizers import Adam


def setup_sequential(y: Union[np.ndarray, int],
                     vocabulary_size,
                     embedding_size=32, embedding_matrix: np.ndarray = None, retrain_matrix: bool = False,
                     additional_layers: List[Layer] = None,
                     ) -> Tuple[Sequential, Any]:
    """
    Train a LSTM model for text classification tasks
    :param y: training labels or the number of training labels
    :param vocabulary_size: size of the input vocabulary
    :param vocabulary_size: size of the input vocabulary
    :param embedding_size: size of the embedding layer (is ignored in case `embedding_matrix` is set)
    :param embedding_matrix: pre-trained embedding matrix to use instead of training new embedding layer
    :param retrain_matrix: continue training the pre-trained embedding matrix (in case `embedding_matrix` is specified)
    :param additional_layers: additional layer definitions downstream the embeddings
    :return: the untrained model and the loss function
    """
    # create a sequential model
    model = Sequential()

    # first layer: embedding
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable=retrain_matrix))
    else:
        model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_size))

    # further layers ...
    if additional_layers:
        for additional_layer in additional_layers:
            model.add(additional_layer)

    # final layer: provides class/label output
    num_categories: int = y.shape[1] if not isinstance(y, int) else y
    if num_categories > 1:
        model.add(Dense(num_categories, activation='softmax'))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'

    model.summary()

    return model, loss


def train_lstm_classifier(x: np.ndarray, y: np.ndarray, vocabulary_size,
                          embedding_size=32, embedding_matrix: np.ndarray = None, retrain_matrix: bool = False,
                          additional_layers: List[Layer] = None,
                          batch_size=32, num_epochs=3, lr=0.001, decay=0.,
                          shuffle_data: Union[int, bool] = False, validation_split: float = 0.,
                          validation_data: Union[Tuple[np.ndarray, np.ndarray],
                                                 Tuple[np.ndarray, np.ndarray, Any]] = None,
                          **kwargs) -> Sequential:
    """
    Train a LSTM model for text classification tasks
    :param x: training data
    :param y: training labels
    :param vocabulary_size: size of the input vocabulary
    :param embedding_size: size of the embedding layer (is ignored in case `embedding_matrix` is set)
    :param embedding_matrix: pre-trained embedding matrix to use instead of training new embedding layer
    :param retrain_matrix: continue training the pre-trained embedding matrix (in case `embedding_matrix` is specified)
    :param additional_layers: additional layer definitions downstream the embeddings
    :param batch_size: Number of samples per gradient update
    :param lr: Learning rate
    :param decay: Learning decay
    :param num_epochs: Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data
    provided.
    :param shuffle_data: shuffle the training to avoid problems in input order (e.g., if data is sorted by label).
    If `shuffle_data=False`, the data will not be shuffled, if `shuffle_data=True`, the data will be shuffled randomly,
    if `shuffle_data` is an integer, this value will be used as seed for the random function.
    :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate
    the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
    `validation_data` will override `validation_split`.
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The
    model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any
    model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the
    `x` and `y` data provided, before shuffling.
    :return: the trained (fitted) model
    """
    assert len(x) == len(y), "The x and y data matrices need to contain the same number of instances!"

    # create a sequential model
    model, loss = setup_sequential(y=y,
                                   vocabulary_size=vocabulary_size,
                                   embedding_size=embedding_size, embedding_matrix=embedding_matrix,
                                   retrain_matrix=retrain_matrix,
                                   additional_layers=additional_layers,
                                   )
    model.compile(loss=loss,
                  optimizer=Adam(lr=lr, decay=decay),
                  metrics=['accuracy'])

    if shuffle_data is not False:
        if shuffle_data is not True:
            # assume `shuffle_training_data` contains the random seed
            np.random.seed(shuffle_data)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
    try:
        # start the training (fit the model to the data)
        model.fit(x=x, y=y,
                  shuffle=False, validation_split=validation_split, validation_data=validation_data,
                  batch_size=batch_size, epochs=num_epochs, **kwargs)
    except KeyboardInterrupt:
        print()
        logging.warning(f"KeyboardInterrupt: Interrupting model fit ...")

    return model

