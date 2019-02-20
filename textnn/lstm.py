from typing import List, Union, Tuple

import numpy as np
from keras import Sequential
from keras.callbacks import History
from keras.engine.base_layer import Layer
from keras.layers import Embedding, LSTM, Dense


def train_lstm_classifier(x: np.ndarray, y: np.ndarray, vocabulary_size,
                          embedding_size=32, embedding_matrix: np.ndarray = None,
                          lstm_layer_size=100, additional_layers: List[Layer] = None,
                          batch_size=32, num_epochs=3, shuffle_data: Union[int, bool] = False,
                          ) -> Tuple[Sequential, History]:
    """
    Train a LSTM model for text classification tasks
    :param x: training data
    :param y: training labels
    :param vocabulary_size: size of the input vocabulary
    :param embedding_size: size of the embedding layer (is ignored in case `embedding_matrix` is set)
    :param embedding_matrix: pre-trained embedding matrix to use instead of training new embedding layer
    :param lstm_layer_size: size of the LSTM layer
    :param additional_layers: additional downstream layer definitions
    :param shuffle_data: shuffle the training to avoid problems in input order (e.g., if data is sorted by label).
    If `shuffle_data=False`, the data will not be shuffled, if `shuffle_data=True`, the data will be shuffled randomly,
    if `shuffle_data` is an integer, this value will be used as seed for the random function.
    :param batch_size: Number of samples per gradient update
    :param num_epochs: Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data
    provided.
    :return: the trained (fitted) model
    """
    assert len(x) == len(y), "The x and y data matrices need to contain the same number of instances!"

    # create a sequential model
    model = Sequential()

    # first layer: embedding
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable=False))
    else:
        model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_size))

    # second layer: LSTM
    model.add(LSTM(lstm_layer_size))

    # further layers ...
    if additional_layers:
        for additional_layer in additional_layers:
            model.add(additional_layer)

    # final layer: provides class/label output
    num_categories = y.shape[1]
    if num_categories > 1:
        model.add(Dense(num_categories, activation='softmax'))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'

    model.summary()

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    if shuffle_data is not False:
        if shuffle_data is not True:
            # assume `shuffle_training_data` contains the random seed
            np.random.seed(shuffle_data)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

    # use first batch as constant validation set
    x_valid, y_valid = x[:batch_size], y[:batch_size]
    # the rest is used for training
    x_train, y_train = x[batch_size:], y[batch_size:]

    # start the training (fit the model to the data)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    return model, history
