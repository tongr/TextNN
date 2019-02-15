import gc
import logging
import pickle
from typing import List, Tuple, Callable

import numpy as np

from textnn.utils.encoding.label import LabelEncoder
from textnn.utils.encoding.text import AbstractTokenEncoder, AbstractEmbeddingMatcher, TokenSequenceEncoder, \
    VectorFileEmbeddingMatcher


def prepare_encoders(storage_folder, training_data: List[Tuple[str, int]],
                     text_enc_init: Callable[[], AbstractTokenEncoder],
                     embedding_matcher: AbstractEmbeddingMatcher = None,
                     text_encoder_name="text_encoder.pickle",
                     label_encoder_name="label_encoder.pickle",
                     ) -> Tuple[AbstractTokenEncoder, LabelEncoder, np.ndarray, np.ndarray]:
    if not storage_folder.exists():
        storage_folder.mkdir(parents=True, exist_ok=True)

    text_encoder_file = storage_folder / text_encoder_name
    label_encoder_file = storage_folder / label_encoder_name

    text_enc: AbstractTokenEncoder = None
    label_enc: LabelEncoder = None
    if text_encoder_file.exists() and label_encoder_file.exists():
        logging.info(f"Loading encoders from files: {text_encoder_file}, {label_encoder_file}")
        with open(text_encoder_file, "rb") as pickle_file:
            text_enc: AbstractTokenEncoder = pickle.load(pickle_file)
        with open(label_encoder_file, "rb") as pickle_file:
            label_enc: LabelEncoder = pickle.load(pickle_file)

    text_list = list(text for text, lab in training_data)
    if not text_enc or not label_enc:
        # extract vocab (from training data)
        text_enc = text_enc_init()
        text_enc.prepare(texts=text_list, show_progress=True)

        # match embeddings
        if embedding_matcher:
            embedding_matcher.reload_embeddings(token_encoder=text_enc, show_progress=True)

        # create label encoder based on training data
        label_enc = LabelEncoder(labeled_data=training_data)

        #
        # serialize data for next time
        #
        with open(text_encoder_file, 'wb') as pickle_file:
            pickle.dump(text_enc, pickle_file)
        with open(label_encoder_file, 'wb') as pickle_file:
            pickle.dump(label_enc, pickle_file)

    # extract data vectors (from training data)
    x_train: np.ndarray = text_enc.encode(texts=text_list)

    # prepare training labels
    y_train: np.ndarray = label_enc.make_categorical(labeled_data=training_data)

    # cleanup memory
    text_list = None
    gc.collect()

    return text_enc, label_enc, x_train, y_train
