import gc
import hashlib
import logging
from pathlib import PurePath, Path
from typing import List, Tuple, Union, Iterable, Any

from keras import Sequential
from keras.layers import *
from keras.models import save_model, load_model
from keras.callbacks import History, CSVLogger
from textnn.lstm import train_lstm_classifier
from textnn.utils import plot2file, join_name, read_text_file, write_text_file
from textnn.utils.encoding import prepare_encoders, LabelEncoder, AbstractTokenEncoder
from textnn.utils.encoding.text import TokenSequenceEncoder, VectorFileEmbeddingMatcher


#
# IMDb specific functions
#
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
    def __init__(self, data_folder, vocabulary_size: int = 4096, max_text_length: int = 512,
                 embedding_size: int = 32,
                 pretrained_embeddings_file=None, embed_reserved: bool = True, retrain_embedding_matrix: bool = False,
                 layer_definitions: str = None,
                 batch_size: int = 1024, num_epochs: int = 25, learning_rate: float = 0.001, learning_decay: float = 0.,
                 shuffle_training_data: Union[int, bool] = 113, validation_split: float = .05,
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
        self._layers, self._layer_definitions = self.parse_layer_definitions(
            layer_definitions if layer_definitions else "Dropout(0.5)|LSTM(8,dropout=0.5)")
        if log_config:
            logging.info(f"{self.__class__.__name__}-configuration:\n{self.config}")

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
    def config(self) -> str:
        return "\n".join(f"  {key}: {value}" for key, value in sorted(self._config_parameters))

    @property
    def _encoder_folder(self) -> Path:
        # name sub-folder
        return self._data_folder / join_name([
            # create name by joining all of the following elements (remove empty strings / None)
            "sequences",
            f"vocab{self._vocabulary_size}",
            f"pad{self._max_text_length}" if self._max_text_length else None,
        ])

    @staticmethod
    def parse_layer_definitions(layer_definitions: Union[str, List[str]], sep="|"):
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
        p = re.compile("^({})\([^(]*\)$".format("|".join(layer_class_names(["keras.layers"]))))
        layers = []
        for position, layer_def in enumerate(layer_definitions):
            m = p.fullmatch(layer_def)
            if not m:
                logging.error(f"Illegal layer definition found in position {position}: \"{layer_def}\"")
                raise ValueError(f"Illegal layer definition found in position {position}: \"{layer_def}\"")

            layers.append(eval(layer_def))
        return layers, layer_definitions

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

    def _train_or_load_model_and_encoders(self, training_data: List[Tuple[str, int]]) -> Tuple[Sequential,
                                                                                               AbstractTokenEncoder,
                                                                                               LabelEncoder]:
        # prepare encoder and encode training data
        self._text_enc, self._label_enc, x_train, y_train = prepare_encoders(
            storage_folder=self._encoder_folder,
            training_data=training_data,
            text_enc_init=lambda: TokenSequenceEncoder(
                limit_vocabulary=self._vocabulary_size,
                default_length=self._max_text_length),
        )
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
        # plot loss
        y_series = {"Training loss": history.history['loss'], }
        if "val_loss" in history.history:
            y_series["Validation loss"] = history.history["val_loss"]
        plot2file(
            file=self._experiment_folder / "loss.png",
            x_values=list(range(self._num_epochs)), y_series=y_series,
            title="Training and validation loss", x_label="Epochs", y_label="Loss",
        )

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
            self._model: Sequential = train_lstm_classifier(
                x=x_train, y=y_train,
                vocabulary_size=self._text_enc.vocabulary_size,
                embedding_size=self._embedding_size,
                embedding_matrix=embedding_matcher.embedding_matrix if embedding_matcher else None,
                retrain_matrix=self._retrain_embedding_matrix,
                additional_layers=self._layers,
                num_epochs=self._num_epochs, batch_size=self._batch_size,
                lr=self._learning_rate, decay=self._learning_decay,
                shuffle_data=self._shuffle_training_data, validation_split=self._validation_split,
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
        logging.info("\n{}".format(classification_report(y_true=y_test, y_pred=y_predicted, target_names=["neg", "pos"],)))
        import json
        write_text_file(
            file_path=self._experiment_folder / "test.json",
            text=json.dumps(classification_report(y_true=y_test,
                                                  y_pred=y_predicted,
                                                  target_names=["neg", "pos"],
                                                  output_dict=True)))

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
