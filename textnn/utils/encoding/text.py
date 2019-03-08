import logging
from abc import ABC, abstractmethod, ABCMeta
from collections import OrderedDict
from pathlib import PurePath, Path
from typing import Iterable, List, Union, Optional, NamedTuple, Tuple, Iterator

import numpy as np
from keras.preprocessing.text import Tokenizer

from textnn.utils import ProgressIterator


class AbstractTextEncoder(ABC):

    def encode(self, texts: Iterable[str], show_progress: bool = True, **kwargs) -> np.ndarray:
        """
        Encode the specified texts based on this encoding strategy
        :param texts: texts to be encoded.
        Please note, the resulting sequences will have the length of `pad_len + 1` because the sequence is preceeded by
        the start_char "^"
        :param show_progress: enables progress feedback for the encoding.
        :return: the feature representation
        """
        if show_progress:
            texts = ProgressIterator(texts, "Encoding text data ...")

        return self._encode(texts, **kwargs)

    @abstractmethod
    def _encode(self, texts: Iterable[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def decode(self, data: np.ndarray, row: int, **kwargs) -> object:
        """
        Decode the given sequence representation based on this encoding strategy.
        :param data: the numpy matrix of the text representation
        :param row: the row (instance) to be decoded
        :return: An object representation the encoded text data.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def decode_all(self, data: np.ndarray, show_progress: bool = True, **kwargs) -> Iterable[object]:
        """
        Decode all given sequence representations based on this encoding strategy.
        :param data: the numpy matrix of the text representations
        :param show_progress: enables progress feedback for the decoding.
        :return: An iterable of object representations the encoded text data.
        """
        index_it = range(data.shape[0])
        if show_progress:
            index_it = ProgressIterator(index_it, "Decoding text data ...")

        return (self.decode(data=data, row=idx, **kwargs) for idx in index_it)

    def decode_str(self, data: np.ndarray, row: int = None, **kwargs) -> Union[List[str], str]:
        """
        Decode the given sequence representation based on this encoding strategy.
        :param data: the numpy matrix of the text representation
        :param row: the row (instance) to be decoded
        :return: A string representation the encoded text data.
        """
        return self._decoded_to_str(self.decode(data=data, row=row, **kwargs))

    def decode_all_str(self, data: np.ndarray, show_progress: bool = True, **kwargs) -> Iterable[str]:
        """
        Decode all given sequence representations based on this encoding strategy.
        :param data: the numpy matrix of the text representations
        :param show_progress: enables progress feedback for the decoding.
        :return: An iterable of string representations the encoded text data.
        """
        return (self._decoded_to_str(obj) for obj in self.decode_all(data=data, show_progress=show_progress, **kwargs))

    def _decoded_to_str(self, decoded_obj: object) -> str:
        return str(decoded_obj)

    @abstractmethod
    def prepare(self, texts: Iterable[str], show_progress: bool = True):
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def name(self) -> str:
        """
        gets the name of this encoder
        """
        return self.__class__.__name__

    def print_representations(self, example_texts: Union[Tuple[str], List[str]], print_function=print, **kwargs):
        """
        Util method to visualize the way the text representations work
        :param example_texts: the texts to be encoded
        :param print_function: function to be used to print the text representation (default: system print)
        :param kwargs: additional parameters to use to encode and decode data (optional)
        """
        seq: np.ndarray = self.encode(example_texts, **kwargs)
        texts_reproduced: List[str] = list(
            self.decode_all_str(seq, show_progress=False, **kwargs))
        for idx, text_reproduced in enumerate(texts_reproduced):
            print_function("{} --> {} ({})".format(example_texts[idx], repr(text_reproduced), seq[idx]))


class AbstractTokenEncoder(AbstractTextEncoder, metaclass=ABCMeta):

    def __init__(self, skip_top_words: int = 0, limit_vocabulary: int = None, reserved_words: list = None, **kwargs):
        """
        Initialize the text encoder

        by convention, use word 0 for padding and 1 as OOV word

        :param skip_top_words: skip the top N most frequently occurring words (which may not be informative).
        :param reserved_words: a list of reserved words in the vocabulary that will be placed with the lowest
        word_indices (default: ["<UNUSED>", "<OOV>"]). Currently, the second element is expected to be the OOV word.
        :param limit_vocabulary: limits the vocabulary size, i.e., max number of words to include. Words are ranked by
        how often they occur (in the training set) and only the most frequent words are kept. If `skip_top_words` is
        used, the least common token in the accepted vocabulary will be `limit_vocab + skip_top_words`.
        """
        # default:
        # unused index 0
        # OOV / unknown / out of vocabulary
        self._reserved_words = reserved_words if reserved_words else ["<UNUSED>", "<OOV>"]
        assert len(self._reserved_words) > 1, "Expecting the second element of `reserved_words` to be the OOV indicator"

        self._skip_top_words = skip_top_words
        assert limit_vocabulary is None or limit_vocabulary > 2, "Keras tokenizer requires 2 reserved words, hence, " \
                                                                 "expecting vocabulary size of over 2!"
        num_words = None if not limit_vocabulary else \
            limit_vocabulary + self._skip_top_words - self._num_reserved_words_unknown_to_tokenizer

        self.tokenizer = Tokenizer(oov_token=self._reserved_words[1], num_words=num_words, **kwargs)
        # by convention, the Tokenizer uses 0 as padding, 1 as OOV word

        # TODO support stop words (instead or in addition to skip top words) or stemming?

    @property
    def _relevant_tokenizer_index_limit(self) -> int:
        return self.tokenizer.num_words if self.tokenizer.num_words else len(self.tokenizer.word_index) + 1

    @property
    def vocabulary_size(self) -> int:
        return self._relevant_tokenizer_index_limit - self._skip_top_words + \
               self._num_reserved_words_unknown_to_tokenizer

    @property
    def _num_reserved_words(self) -> int:
        return len(self._reserved_words)

    def is_reserved_index(self, word_index) -> bool:
        return word_index < self._num_reserved_words

    @property
    def _num_reserved_words_unknown_to_tokenizer(self) -> int:
        # the tokenizer reserves the first two word indexes:
        # unused index 0
        # OOV / unknown / out of vocabulary 1
        return max(0, self._num_reserved_words - 2)

    @property
    def oov_token(self) -> str:
        return self.tokenizer.oov_token

    @property
    def oov_token_index(self) -> int:
        return self.tokenizer.word_index.get(self.oov_token)

    def prepare(self, texts: Iterable[str], show_progress: bool = True) -> Tokenizer:
        if show_progress:
            texts = ProgressIterator(texts, "Preparing vocabulary ...")

        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer

    def _is_tokenizer_index_included(self, tokenizer_index):
        assert tokenizer_index >= 0

        # reserved indices are always accepted:
        # unused index 0
        # OOV / unknown / out of vocabulary 1
        if tokenizer_index < 2:
            return True

        # we need to remove the "wrong" word indices from the tokenizer somewhere else
        # if the word index is too high, we won't accept it
        if tokenizer_index >= self._relevant_tokenizer_index_limit:
            return False

        # we want to remove the most frequent words .. check if it is one of them
        if self._skip_top_words and tokenizer_index < self._skip_top_words + 2:
            return False

        return True

    def index_to_word(self, word_index: int) -> str:
        if not 0 <= word_index < self.vocabulary_size:
            return self.oov_token

        if self.is_reserved_index(word_index):
            return self._reserved_words[word_index]

        tokenizer_index = word_index - self._num_reserved_words_unknown_to_tokenizer + self._skip_top_words

        return self.tokenizer.index_word.get(tokenizer_index, self.oov_token)

    def word_to_index(self, word: str) -> int:
        try:
            reserved_word_index = self._reserved_words.index(word)
            # here we are handling the special token that might indicate starting/stopping/etc.
            return reserved_word_index
        except ValueError:
            pass
        tokenizer_index = self.tokenizer.word_index.get(word)

        if tokenizer_index is not None:
            return self._tokenizer_to_word_index(tokenizer_index)

        return self.oov_token_index

    def _tokenizer_to_word_index(self, tokenizer_index: int) -> int:
        if tokenizer_index < 2:
            # the tokenizer reserves the first two word indexes:
            # unused index 0
            # OOV / unknown / out of vocabulary 1
            return tokenizer_index

        if not self._is_tokenizer_index_included(tokenizer_index):
            return self.oov_token_index

        word_index = tokenizer_index
        if self._num_reserved_words_unknown_to_tokenizer > 0 or self._skip_top_words:
            # in case we have further additional reserved words (more than two that are already covered in the tokenizer
            # instance) or have to skip top words, we have to transform these indices
            word_index = tokenizer_index + self._num_reserved_words_unknown_to_tokenizer - self._skip_top_words

        return word_index

    def encode(self, texts: Iterable[str], **kwargs):
        assert self.tokenizer.document_count > 0, "TextEncoder has to be initialized using prepare(...)"
        return super().encode(texts=texts, **kwargs)


class BowEncoder(AbstractTokenEncoder):
    def __init__(self, skip_top_words: int = 0, limit_vocabulary: int = None, mode: str = "count", **kwargs):
        """
        Initialize the Bag-of-Word text encoder

        by convention, use 1 as OOV word

        all characters: 0 (padding), 2 (OOV)
        :param skip_top_words: skip the top N most frequently occurring words (which may not be informative).
        :param limit_vocabulary: limits the vocabulary size, i.e., max number of words to include. Words are ranked by
        how often they occur (in the training set) and only the most frequent words are kept. If `skip_top_words` is
        used, the least common token in the accepted vocabulary will be `limit_vocab + skip_top_words`.
        :param mode: the selected mode for the BOW weight, one of "binary", "count", "tfidf", "freq"
        """
        super().__init__(skip_top_words, limit_vocabulary, **kwargs)
        self.bow_mode = mode

    def _encode(self, texts: Iterable[str], **kwargs) -> np.ndarray:
        """
        encode the given texts to a numpy matrix according to self.bow_mode
        :param texts: the texts to be encoded
        :return: the numpy matrix of the set of word representation
        """
        word_matrix = self.tokenizer.texts_to_matrix(texts=texts, mode=self.bow_mode)

        # get a filter for not included columns
        num_col = word_matrix.shape[1]
        col_filter = list(x for x in range(num_col) if self._is_tokenizer_index_included(x))

        # wind the removed token columns and increase column of self.oov_token_index accordingly
        if num_col != len(col_filter):
            # count all words removed (not included columns) and increase OOV accordingly
            inverse_col_filter = list({x for x in range(num_col)} - set(col_filter))
            add_oov_per_doc = np.take(word_matrix, inverse_col_filter, axis=1).sum(axis=1)
            if self.bow_mode == 'binary':
                # we need to normalize the update vector, such that word matrix does only contain 0s and 1s
                # find out where oov is not yet set and should be set
                add_oov_per_doc = np.logical_and(add_oov_per_doc,
                                                 np.logical_xor(word_matrix[:, 1], add_oov_per_doc)
                                                 ).astype(float)

            if np.sum(add_oov_per_doc):
                add_oov_matrix = np.zeros(word_matrix.shape)
                add_oov_matrix[:, 1] = add_oov_per_doc

                word_matrix = word_matrix + add_oov_matrix

            # remove elements not in col_filter
            word_matrix = np.take(word_matrix, col_filter, axis=1)
            # consider removing column zero: unused index 0

        return word_matrix

    def decode(self, data: np.ndarray, row: int, ignore_zero_freq=False, **kwargs) -> OrderedDict:
        """
        Decode the given BOW representation based on this vocabulary.
        :param data: the numpy matrix of the BOW representation
        :param row: the row (instance) to be decoded
        :param ignore_zero_freq: remove entries from the bag that have a zero frequency.
        :return: A mapping of words to frequencies representing the BOW.
        """
        assert np.size(data, 0) > row >= 0, "Illegal row index found"
        word_bag_vector: np.ndarray = data[row]
        word_bag_index = 0
        bag_representation = OrderedDict()
        for word_index in range(self.vocabulary_size):
            if word_bag_vector.size <= word_bag_index:
                break
            value = word_bag_vector[word_bag_index]
            word_bag_index += 1
            if not ignore_zero_freq or value != 0:
                bag_representation[self.index_to_word(word_index)] = value

        return bag_representation

    @property
    def name(self) -> str:
        """
        gets the name of this encoder
        """
        return "{}(mode={})".format(BowEncoder.__name__, self.bow_mode)

    def _decoded_to_str(self, decoded_obj: OrderedDict) -> str:
        return ", ".join(repr(x) for x in decoded_obj.items())


class TokenSequenceEncoder(AbstractTokenEncoder):
    @property
    def padding_token(self) -> str:
        return self._reserved_words[self.padding_token_index]

    @property
    def padding_token_index(self) -> int:
        return 0

    @property
    def start_token(self) -> str:
        return self._reserved_words[self.start_token_index]

    @property
    def start_token_index(self) -> int:
        return 2

    @property
    def end_token(self) -> str:
        return self._reserved_words[self.end_token_index]

    @property
    def end_token_index(self) -> int:
        return 3

    def __init__(self, default_length: int = None, skip_top_words: int = 0, limit_vocabulary: int = None,
                 add_start_end_indicators: bool = True, pad_beginning: bool = True,
                 **kwargs):
        """
        Initialize the Bag-of-Word text encoder

        by convention, use 1 as OOV word

        all characters: 0 (padding), 2 (OOV)
        :param default_length: length to which input text will be normalized per default (i.e., sequences longer than
        this will be trimmed, shorter sequences will be padded). If negative, the text will be trimmed at the start of
        the string such that so that the last `default_length` token of the text are encoded. If positive, the end of
        long strings is removed if necessary.
        :param skip_top_words: skip the top N most frequently occurring words (which may not be informative).
        :param limit_vocabulary: limits the vocabulary size, i.e., max number of words to include. Words are ranked by
        how often they occur (in the training set) and only the most frequent words are kept. If `skip_top_words` is
        used, the least common token in the accepted vocabulary will be `limit_vocab + skip_top_words`.
        :param add_start_end_indicators: if True, add `"<START>"` and `"<END>"` indicators at start and end of the
        encoded sequences
        :param pad_beginning: if True, add padding (of short strings) at the start of the sequence, otherwise pad the
        end
        :param mode: the selected mode for the BOW weight, one of "binary", "count", "tfidf", "freq"
        """
        # by convention, the Tokenizer uses 0 as padding, 1 as OOV word, hence, we use the following setup
        # 0 - PAD
        # 1 - OOV / unknown / out of vocabulary
        # 2 - START / start_token / start token
        # 3 - END / end_token / end token
        super().__init__(skip_top_words, limit_vocabulary,
                         reserved_words=["<PAD>", "<OOV>", "<START>", "<END>"],
                         **kwargs)
        self.default_length = default_length
        self.add_start_end_indicators = add_start_end_indicators
        self.pad_beginning = pad_beginning

    def _texts_to_normalized_sequences(self, texts) -> Iterable[List[int]]:
        return ([self._tokenizer_to_word_index(x) for x in text_vec] for text_vec in
                self.tokenizer.texts_to_sequences_generator(texts))

    def _encode(self, texts: Iterable[str], length: int = None, **kwargs) -> np.ndarray:
        """
        Encode the specified texts based on the prepared vocabulary
        :param texts: texts to be encoded.
        Please note, the resulting sequences will have the length of `pad_len + 1` because the sequence is preceeded by
        the start_char '^' or the oov token ('*')
        :param length: length to which input text will be normalized (i.e., sequences longer than this
        will be trimmed, shorter sequences will be padded). If negative, the text will be trimmed at the start of
        the string such that so that the last `default_length` token of the text are encoded. If positive, the end of
        long strings is removed if necessary.
        :return: the sequence representation of this
        """
        max_token: int = None
        trim_end = True
        if length is not None:
            # decrease for in order to account `start_char`/`end_char`
            max_token: int = abs(length)
            trim_end = length > 0
        elif self.default_length is not None:
            # decrease for in order to account `start_char`/`end_char`
            max_token: int = abs(self.default_length)
            trim_end = self.default_length > 0

        if max_token and self.add_start_end_indicators:
            max_token -= 2

        max_seq_len = 0
        xs = []
        for encoded_text in self._texts_to_normalized_sequences(texts):
            xs.append(encoded_text)
            if max_seq_len < len(encoded_text):
                max_seq_len = len(encoded_text)

        # now we have to produce a feature vector with:
        # start token .. actual encoded data .. filled up until `max_len` is reached

        if max_token is None:
            max_token: int = max_seq_len

        for idx, x in enumerate(xs):
            # prepare padding
            padding = [self.padding_token_index] * (max_token - len(x))
            # trim x to max size (from beginning or end
            x = x[:max_token] if trim_end else x[-max_token:]
            if self.add_start_end_indicators:
                # use start/end indicators
                x.insert(0, self.start_token_index)
                x.append(self.end_token_index)

            if padding:
                # padding at start or end
                x = (padding + x) if self.pad_beginning else (x + padding)

            xs[idx] = x

        return np.array(xs)

    def decode(self, data: np.ndarray, row: int, show_padding: bool = None, show_start_end: bool = None, **kwargs) -> \
            List[str]:
        """
        Decode the given sequence representation based on this vocabulary.
        :param data: the numpy matrix of the sequence representation
        :param row: the row (instance) to be decoded
        :param show_padding: if True, padding will be represented in the sequence.
        :param show_start_end: if True, `start_char`/`end_char` will be represented in the sequence.
        :return: A list of token representing.
        """
        if show_padding is None:
            show_padding = self.default_length is not None
        if show_start_end is None:
            show_start_end = self.add_start_end_indicators
        assert np.size(data, 0) > row >= 0, "Illegal row index found"
        id_sequence = data[row]
        unpadded_sequence = np.trim_zeros(id_sequence, "f" if self.pad_beginning else "b")

        assert not self.add_start_end_indicators or unpadded_sequence[0] == self.start_token_index, \
            "Unexpected start_char '{0[0]}' found in {1}!".format(unpadded_sequence, id_sequence)
        assert not self.add_start_end_indicators or unpadded_sequence[-1] == self.end_token_index, \
            "Unexpected end_char '{0[-1]}' found in {1}!".format(unpadded_sequence, id_sequence)

        if not show_start_end and self.add_start_end_indicators:
            num_padding = len(id_sequence) - len(unpadded_sequence)
            unpadded_sequence = np.delete(unpadded_sequence, [0, len(unpadded_sequence)-1])
            id_sequence = [0]*num_padding + unpadded_sequence.tolist()

        if not show_padding:
            id_sequence = unpadded_sequence

        return list(self.index_to_word(i) for i in id_sequence)

    def _decoded_to_str(self, decoded_obj: List[str]) -> str:
        return " ".join(decoded_obj)


class WordVector(NamedTuple):
    word: str
    values: List[str]

    def vector(self, vector_length: int) -> np.ndarray:
        return np.fromiter(map(float, self.values[:vector_length]), dtype=np.float)


class AbstractEmbeddingMatcher(ABC):
    def __init__(self, encode_reserved_words: bool = False):
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_length: int = None
        self.encode_reserved_words = encode_reserved_words
        self.normalize_reserved_embeddings_by: float = 1e-16

    @abstractmethod
    def get_vector_source(self) -> Tuple[int, int, Iterable[WordVector]]:
        raise NotImplementedError("Subclasses should implement this!")

    def reload_embeddings(self, token_encoder: AbstractTokenEncoder, show_progress=False):
        num_embeddings, embedding_length, vectors = self.get_vector_source()

        if show_progress:
            vectors = ProgressIterator(vectors, "Initializing embedding matrix ...", target=num_embeddings, interval=5)

        # fixed: embedding_matrix = np.zeros([self._skip_top_words + self.vocabulary_size, embedding_length])
        # remove skip words from embedding matrix
        embedding_matrix = np.zeros([token_encoder.vocabulary_size, embedding_length])
        found_word_indices = set()
        words_not_in_vocab_sample = []
        # add embeddings for normal words
        for word_vec in vectors:
            word_index = token_encoder.word_to_index(word_vec.word)
            if not token_encoder.is_reserved_index(word_index):
                embedding_matrix[word_index] = word_vec.vector(vector_length=embedding_length)
                found_word_indices.add(word_index)
            elif len(words_not_in_vocab_sample) < 5000:
                words_not_in_vocab_sample.append(word_vec.word)

        if self.encode_reserved_words:
            # in case we encode reserved words specifically (not all by zero-vectors), we need to update the affected
            # rows of the embedding matrix
            for row_index in range(embedding_matrix.shape[0]):
                if token_encoder.is_reserved_index(row_index):
                    # instead of leaving the reserved words as zero vectors, we create special embeddings for them
                    embedding_matrix[row_index] = self._reserved_word_embedding(row_index, embedding_length)
                elif row_index not in found_word_indices:
                    # all words not assigned in the embedding matrix should be replaced with OOV
                    embedding_matrix[row_index] = embedding_matrix[token_encoder.oov_token_index]

        self._embedding_matrix: Optional[np.ndarray] = embedding_matrix
        self._embedding_length: int = embedding_length

        logging.debug("Sample words not found: {}".format(np.random.choice(words_not_in_vocab_sample, 50)))

    def _reserved_word_embedding(self, reserved_index: int, embedding_length: int):
        # add embeddings for reserved token and introduce "artificial" embeddings
        # (e.g., [1,1,1,...], [-1,-1,-1,...], ...) to encode the reserved token
        predefined_pattern = [
            # 0 - PAD
            [0],
            # 1 - OOV / unknown / out of vocabulary
            [1.],
            # 2 - START / start_token / start token
            [-1.],
            # 3 - END / end_token / end token
            [1., -1.],
            # ... future special token
            [-1, 1.],
            [1., 1., -1.], [1., -1., 1.], [1., -1., -1.], [-1., 1., 1.], [-1., 1., -1.], [-1., -1., 1.],
        ]

        pattern = predefined_pattern[reserved_index]
        artificial_vec = (pattern * int(embedding_length / len(pattern) + 1))[:embedding_length]
        return np.fromiter(map(float, artificial_vec), dtype=np.float) * self.normalize_reserved_embeddings_by

    @property
    def embedding_matrix(self) -> np.ndarray:
        if self._embedding_matrix is None:
            raise ValueError("Embeddings not yet loaded, matrix not yet initialized!")
        return self._embedding_matrix


class VectorFileEmbeddingMatcher(AbstractEmbeddingMatcher):

    def __init__(self, fasttext_vector_file: Union[str, Path], encode_reserved_words: bool = False):
        super().__init__(encode_reserved_words=encode_reserved_words)
        assert fasttext_vector_file, "Valid fasttext_vector_file required!"
        if not isinstance(fasttext_vector_file, PurePath):
            fasttext_vector_file = Path(fasttext_vector_file)

        self._fasttext_vector_file: Path = fasttext_vector_file
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_length: int = None

    @staticmethod
    def _parse_vector_file(vector_filename) -> Iterator[WordVector]:
        with open(vector_filename, encoding='utf-8') as vector_file:
            for line in vector_file:
                values = line.rstrip().rsplit(' ')
                yield WordVector(word=values[0], values=values[1:])

    def get_vector_source(self) -> Tuple[int, int, Iterable[WordVector]]:
        assert self._fasttext_vector_file.exists(), "Fast text vector file '{}' not found!".format(
            self._fasttext_vector_file)

        vectors = self._parse_vector_file(str(self._fasttext_vector_file))

        # read header line of vector file
        first_entry: WordVector = next(vectors)
        num_embeddings = int(first_entry.word)
        embedding_length = int(first_entry.values[0])

        return num_embeddings, embedding_length, vectors


def print_all_representations(example_texts: List[str],
                              encs: Union[AbstractTextEncoder, Iterable[AbstractTextEncoder]] = None,
                              print_function=print,
                              **kwargs):
    """
    Util method to show the different kinds of text representations
    :param example_texts: the texts to be encoded
    :param encs: list of encoders to use (optional)
    :param print_function: function to be used to print the text representation (default: system print)
    :param kwargs: parameters to use to create encoders, in case `encs` is None (optional)
    """
    if encs is None:
        encs = [
            TokenSequenceEncoder(**kwargs),
            BowEncoder(mode="count", **kwargs),
            BowEncoder(mode="binary", **kwargs),
            BowEncoder(mode="freq", **kwargs),
            BowEncoder(mode="tfidf", **kwargs)
        ]

        for enc in ProgressIterator(encs, "Preparing encoders ..."):
            enc.prepare_vocabulary(texts=example_texts, show_progress=False)

    if not isinstance(encs, Iterable):
        encs = [encs]

    for enc in encs:
        print("Encoding based on {}:".format(enc.name))
        enc.print_representations(example_texts)
