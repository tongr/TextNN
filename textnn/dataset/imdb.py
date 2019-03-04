import gc
import hashlib
import logging
import pickle
from pathlib import PurePath, Path
from typing import List, Tuple, Union, Iterable, Any

from keras import Sequential
from keras.callbacks import History, CSVLogger
from keras.layers import *
from keras.models import save_model, load_model
from keras.optimizers import Adam

from textnn.utils import plot2file, join_name, read_text_file, write_text_file
from textnn.utils.encoding.label import LabelEncoder
from textnn.utils.encoding.text import AbstractTokenEncoder, TokenSequenceEncoder, VectorFileEmbeddingMatcher


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


class ImdbClassifier:
    def __init__(self, data_folder, vocabulary_size: int = 4096, max_text_length: int = 512,
                 embedding_size: int = 32,
                 pretrained_embeddings_file=None, embed_reserved: bool = True, retrain_embedding_matrix: bool = False,
                 layer_definitions: str = None,
                 batch_size: int = 1024, num_epochs: int = 25, learning_rate: float = 0.001, learning_decay: float = 0.,
                 shuffle_training_data: Union[int, bool] = 113, validation_split: float = .05,
                 log_config: bool = True,
                 ):
        """

        :param data_folder:
        :param vocabulary_size: size of the input vocabulary
        :param max_text_length:
        :param embedding_size: size of the embedding layer (is ignored in case `embedding_matrix` is set)
        :param pretrained_embeddings_file:
        :param embed_reserved:
        :param retrain_embedding_matrix: continue training the pre-trained embedding matrix (in case `embedding_matrix`
        is specified)
        :param layer_definitions: additional layer definitions downstream the embeddings
        :param batch_size: Number of samples per gradient update
        :param learning_rate: Learning rate
        :param learning_decay: Learning decay
        :param num_epochs: Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y`
        data provided.
        :param shuffle_training_data: shuffle the training to avoid problems in input order (e.g., if data is sorted by
        label). If `shuffle_data=False`, the data will not be shuffled, if `shuffle_data=True`, the data will be
        shuffled randomly, if `shuffle_data` is an integer, this value will be used as seed for the random function.
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The
        model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and
        any model metrics on this data at the end of each epoch. The validation data is selected from the last samples
        in the `x` and `y` data provided, before shuffling.
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The
        model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and
        any model metrics on this data at the end of each epoch. The validation data is selected from the last samples
        in the `x` and `y` data provided, before shuffling.
        :param log_config: if True a the config of this instance is printed after setup
        """
        self._data_folder: Path = data_folder if isinstance(data_folder, PurePath) else Path(data_folder)
        self._vocabulary_size = vocabulary_size
        self._max_text_length = max_text_length
        self._embedding_size = embedding_size
        if pretrained_embeddings_file:
            pretrained_embeddings_file = pretrained_embeddings_file if isinstance(pretrained_embeddings_file, PurePath)\
                else Path(pretrained_embeddings_file)
        self._pretrained_embeddings_file: Path = pretrained_embeddings_file
        self._retrain_embedding_matrix: bool = retrain_embedding_matrix
        self._embed_reserved = embed_reserved
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._learning_decay = learning_decay
        self._shuffle_training_data = shuffle_training_data
        self._validation_split = validation_split

        self._model: Sequential = None
        self._text_enc: AbstractTokenEncoder = None
        self._label_enc: LabelEncoder = None
        self._layers, self._layer_definitions = self._parse_layer_definitions(
            layer_definitions if layer_definitions else "Dropout(0.5)|LSTM(8,dropout=0.5)")
        if log_config:
            logging.info(f"{self.__class__.__name__}-configuration:\n{self.config}")

    #
    # public methods
    #
    def train_and_evaluate(self):
        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._data_folder, train_only=True))
        # prepare the model
        self._train_or_load_model_and_encoders(training_data)

        del training_data
        gc.collect()

        # extract test data
        test_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._data_folder, train_only=False))
        # evaluate the performance of the model
        self._evaluate_model(test_data)

        del test_data
        gc.collect()

    def test_encoding(self, *texts: str):
        if len(texts) <= 0:
            logging.warning("Please specify at least one text to encode!")
            return

        # get training data
        training_data: List[Tuple[str, int]] = list(imdb_data_generator(base_folder=self._data_folder, train_only=True))

        # prepare the model
        self._train_or_load_model_and_encoders(training_data)

        # debug the text encoder
        logging.info(f"Trying to encode the following texts: {texts}")
        self._text_enc.print_representations(texts)

    @property
    def config(self) -> str:
        return "\n".join(f"  {key}: {value}" for key, value in sorted(self._config_parameters))

    #
    # config accessors
    #
    @staticmethod
    def _parse_layer_definitions(layer_definitions: Union[str, List[str]], sep="|"):
        def layer_class_names(packages):
            import inspect
            import importlib
            classes = set()
            for package in packages:
                module = importlib.import_module(package)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    classes.add(obj.__name__)
            return classes

        if sep:
            layer_definitions = layer_definitions.split(sep)

        import re
        p = re.compile("^({})\\(.*\\)$".format("|".join(layer_class_names(["keras.layers"]))))
        layers = []
        for position, layer_def in enumerate(layer_definitions):
            m = p.fullmatch(layer_def)
            if not m:
                logging.error(f"Illegal layer definition found in position {position}: \"{layer_def}\"")
                raise ValueError(f"Illegal layer definition found in position {position}: \"{layer_def}\"")

            layers.append(eval(layer_def))
        return layers, layer_definitions

    @property
    def _config_parameters(self) -> Iterable[Tuple[str, Any]]:
        from itertools import chain
        kv = chain(
            self.__dict__.items(),
            [("_encoder_folder", self._encoder_folder),
             ("_model_folder", self._model_folder),
             ("_experiment_folder", self._experiment_folder)])
        return ((key.lstrip("_"), value) for key, value in kv)

    @property
    def _encoder_folder(self) -> Path:
        # name sub-folder
        return self._data_folder / join_name([
            # create name by joining all of the following elements (remove empty strings / None)
            "sequences",
            f"vocab{self._vocabulary_size}",
            f"pad{self._max_text_length}" if self._max_text_length else None,
        ])

    @property
    def _model_folder(self) -> Path:
        # name sub-folder
        return self._encoder_folder / join_name([
            # create name by joining all of the following elements (remove empty strings / None)
            "sequential",
            f"emb{self._embedding_size}" if not self._pretrained_embeddings_file else "pretrained-embeddings-{}".format(
                hashlib.md5(open(str(self._pretrained_embeddings_file), "rb").read()).hexdigest()),
            f"retrained" if self._pretrained_embeddings_file and self._retrain_embedding_matrix else None,
            "embed-reserved" if self._pretrained_embeddings_file and self._embed_reserved else None,
            "layers-{}".format("-".join(self._layer_definitions)),
        ])

    @property
    def _experiment_folder(self) -> Path:
        # name sub-folder
        return self._model_folder / join_name([
            # create name by joining all of the following elements (remove empty strings / None)
            "experiment",
            f"epochs{self._num_epochs}",
            f"batch{self._batch_size}",
            f"lr{self._learning_rate}",
            f"decay{self._learning_decay}" if self._learning_decay else None,
            None if self._shuffle_training_data is False else "shuffle" if self._shuffle_training_data is True
            else f"shuffle{self._shuffle_training_data}",
            f"validation-split{self._validation_split}" if self._validation_split else None,
        ])

    #
    # preparation
    #
    # prepare encoders
    def _prepare_or_load_encoders(self,
                                  training_data: List[Tuple[str, int]],
                                  initialized_text_enc: AbstractTokenEncoder,
                                  ) -> Tuple[AbstractTokenEncoder, LabelEncoder]:
        if not self._encoder_folder.exists():
            self._encoder_folder.mkdir(parents=True, exist_ok=True)

        text_encoder_file = self._encoder_folder / "text-encoder.pickle"
        label_encoder_file = self._encoder_folder / "label-encoder.pickle"

        text_enc: AbstractTokenEncoder = None
        label_enc: LabelEncoder = None
        if text_encoder_file.exists() and label_encoder_file.exists():
            logging.info(f"Loading encoders from files: {text_encoder_file}, {label_encoder_file}")
            with open(str(text_encoder_file), "rb") as pickle_file:
                text_enc: AbstractTokenEncoder = pickle.load(pickle_file)
            with open(str(label_encoder_file), "rb") as pickle_file:
                label_enc: LabelEncoder = pickle.load(pickle_file)

        text_list = list(tex for tex, lab in training_data)
        if not text_enc or not label_enc:
            # extract vocab (from training data)
            text_enc = initialized_text_enc
            text_enc.prepare(texts=text_list, show_progress=True)

            # create label encoder based on training data
            label_enc = LabelEncoder(labeled_data=training_data)

            #
            # serialize data for next time
            #
            with open(str(text_encoder_file), 'wb') as pickle_file:
                pickle.dump(text_enc, pickle_file)
            with open(str(label_encoder_file), 'wb') as pickle_file:
                pickle.dump(label_enc, pickle_file)

        # cleanup memory
        del text_list
        gc.collect()

        return text_enc, label_enc

    #
    # prepare model
    #
    def _train_or_load_model_and_encoders(self, training_data: List[Tuple[str, int]]) -> Tuple[Sequential,
                                                                                               AbstractTokenEncoder,
                                                                                               LabelEncoder]:
        # prepare encoder and encode training data
        self._text_enc, self._label_enc = self._prepare_or_load_encoders(
            training_data=training_data,
            initialized_text_enc=TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length),
        )

        # extract data vectors (from training data)
        text_list = list(tex for tex, lab in training_data)
        x_train: np.ndarray = self._text_enc.encode(texts=text_list)

        # cleanup memory
        del text_list
        gc.collect()

        # prepare training labels
        y_train: np.ndarray = self._label_enc.make_categorical(labeled_data=training_data)

        # load or train model
        self._train_or_load_model(x_train, y_train)
        return self._model, self._text_enc, self._label_enc

    def _plot_training_stats(self, history: History):
        # plot accuracy
        y_series = {"Training acc": history.history["acc"], }
        if "val_acc" in history.history:
            y_series["Validation acc"] = history.history["val_acc"]
        plot2file(
            file=self._experiment_folder / "accuracy.png",
            x_values=list(range(self._num_epochs)), y_series=y_series,
            title="Training and validation accuracy", x_label="Epochs", y_label="Accuracy",
        )
        # plot MSE
        y_series = {"Training MSE": history.history["mean_squared_error"], }
        if "val_mean_squared_error" in history.history:
            y_series["Validation MSE"] = history.history["val_mean_squared_error"]
        plot2file(
            file=self._experiment_folder / "mse.png",
            x_values=list(range(self._num_epochs)), y_series=y_series,
            title="Training and validation Mean Squared Error", x_label="Epochs", y_label="MSE",
        )
        # plot KL Divergence
        y_series = {"Training KL Divergence": history.history["kullback_leibler_divergence"], }
        if "val_kullback_leibler_divergence" in history.history:
            y_series["Validation Kullback Leibler Divergence"] = history.history["val_kullback_leibler_divergence"]
        plot2file(
            file=self._experiment_folder / "kld.png",
            x_values=list(range(self._num_epochs)), y_series=y_series,
            title="Training and validation KL Divergence", x_label="Epochs", y_label="KL Divergence",
        )
        # plot loss
        y_series = {"Training loss": history.history['loss'], }
        if "val_loss" in history.history:
            y_series["Validation loss"] = history.history["val_loss"]
        plot2file(
            file=self._experiment_folder / "loss.png",
            x_values=list(range(self._num_epochs)), y_series=y_series,
            title="Training and validation loss", x_label="Epochs", y_label="Loss",
        )

    # noinspection PyUnresolvedReferences
    def _train_or_load_model(self, x_train: np.ndarray, y_train: np.ndarray) -> Sequential:
        model_file = self._experiment_folder / "keras-model.hd5"
        if model_file.exists():
            logging.info(f"Loading models from: {model_file}")
            self._model: Sequential = load_model(str(model_file))
        else:
            # this model is inspired by the configuration of Susan Li:
            # https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e

            # load encoders and encode training data
            embedding_matcher = None
            if self._pretrained_embeddings_file and self._pretrained_embeddings_file.exists():
                embedding_matcher = VectorFileEmbeddingMatcher(fasttext_vector_file=self._pretrained_embeddings_file,
                                                               encode_reserved_words=self._embed_reserved,
                                                               )
                # match embeddings with text/token encoder
                embedding_matcher.reload_embeddings(token_encoder=self._text_enc, show_progress=True)

            if not self._experiment_folder.exists():
                self._experiment_folder.mkdir(parents=True, exist_ok=True)
            csv_log = self._experiment_folder / "epoch_results.csv"
            # train the model
            self._model: Sequential = self._train_model(
                x=x_train, y=y_train,
                embedding_matrix=embedding_matcher.embedding_matrix if embedding_matcher else None,
                callbacks=[CSVLogger(str(csv_log), append=True, separator=';')]
            )

            self._plot_training_stats(self._model.history)
            del embedding_matcher
            gc.collect()

            if self._num_epochs <= len(self._model.history.history['loss']):
                # if the training finished: serialize data for next time
                save_model(self._model, filepath=str(model_file))

        self._model.summary()

        return self._model

    def _train_model(self,
                     x: np.ndarray, y: np.ndarray,
                     embedding_matrix: np.ndarray = None,
                     validation_data: Union[Tuple[np.ndarray, np.ndarray],
                                            Tuple[np.ndarray, np.ndarray, Any]] = None,
                     **kwargs) -> Sequential:
        """
        Train a `Sequential` model for text classification tasks
        :param x: training data
        :param y: training labels
        :param embedding_matrix: pre-trained embedding matrix to use instead of training new embedding layer
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to
        evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
        `validation_data` will override `validation_split`.
        :return: the trained (fitted) model
        """
        assert len(x) == len(y), "The x and y data matrices need to contain the same number of instances!"

        # create a sequential model
        model, loss = self._setup_model(y=y, embedding_matrix=embedding_matrix)
        model.compile(loss=loss,
                      optimizer=Adam(lr=self._learning_rate, decay=self._learning_decay),
                      metrics=['accuracy', 'mean_squared_error', 'kullback_leibler_divergence'],
                      )

        if self._shuffle_training_data is not False:
            if self._shuffle_training_data is not True:
                # assume `shuffle_training_data` contains the random seed
                np.random.seed(self._shuffle_training_data)
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
        try:
            # start the training (fit the model to the data)
            model.fit(x=x, y=y,
                      shuffle=False, validation_split=self._validation_split, validation_data=validation_data,
                      batch_size=self._batch_size, epochs=self._num_epochs, **kwargs)
        except KeyboardInterrupt:
            print()
            logging.warning(f"KeyboardInterrupt: Interrupting model fit ...")

        return model

    def _setup_model(self,
                     y: Union[np.ndarray, int],
                     embedding_matrix: np.ndarray = None,
                     ) -> Tuple[Sequential, Any]:
        """
        Prepare a `Sequential` model for text classification tasks
        :param y: training labels or the number of training labels
        :param embedding_matrix: pre-trained embedding matrix to use instead of training new embedding layer
        :return: the untrained model and the loss function
        """
        # create a sequential model
        model = Sequential()

        # first layer: embedding
        if embedding_matrix is not None:
            model.add(Embedding(input_dim=self._vocabulary_size, output_dim=embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=self._retrain_embedding_matrix))
        else:
            model.add(Embedding(input_dim=self._vocabulary_size, output_dim=self._embedding_size))

        # further layers ...
        if self._layers:
            for layer in self._layers:
                model.add(layer)

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

    #
    # evaluation
    #
    def _evaluate_model(self, test_data: List[Tuple[str, int]]):
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

        from sklearn.metrics.classification import classification_report
        logging.info("\n{}".format(classification_report(y_true=y_test,
                                                         y_pred=y_predicted,
                                                         target_names=["neg", "pos"],
                                                         )))
        import json
        write_text_file(
            file_path=self._experiment_folder / "test.json",
            text=json.dumps(classification_report(y_true=y_test,
                                                  y_pred=y_predicted,
                                                  target_names=["neg", "pos"],
                                                  output_dict=True)))
