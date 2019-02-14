from textnn.utils.encoding.text import *

from pytest import approx, raises

# texts from https://en.wikipedia.org/wiki/Python_(programming_language)

corpus = [
    "Python is an interpreted, high-level, general-purpose programming language.",
    "Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code "
    "readability, notably using significant whitespace.",
    "It provides constructs that enable clear programming on both small and large scales.",
    "Van Rossum led the language community until stepping down as leader in July 2018.",
    "Python features a dynamic type system and automatic memory management.",
    "It supports multiple programming paradigms, including object-oriented, imperative, functional and procedural, and "
    "has a large and comprehensive standard library.",
    "Python interpreters are available for many operating systems.",
    "CPython, the reference implementation of Python, is open source software and has a community-based development "
    "model, as do nearly all of Python's other implementations.",
    "Python and CPython are managed by the non-profit Python Software Foundation.",
]
test_sentence = "Python is a multi-paradigm programming language."


def test_sow_encoder_default():
    encoder = BowEncoder(mode="binary")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 100 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 100)
    # test sentence contains 7 words, but because OOV occurs twice and reduces to a binary value of 1, it sums up to 6
    assert np.sum(encoded_test_sentences) == 6
    # two of them are OOV (multi and paradigm)
    assert encoded_test_sentences[0, 1] == 1
    # python occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == 1
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == 1
    # a occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == 1

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 1,
        "python": 1,
        "is": 1,
        "a": 1,
        "programming": 1,
        "language": 1,
    }


def test_sow_encoder_limit_vocab():
    # build a vocab of size 8 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)

    encoder = BowEncoder(limit_vocabulary=8, mode="binary")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 8 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 8)
    # test sentence contains 7 words, but because OOV occurs 4 times and reduces to a binary value of 1, it sums up to 4
    assert np.sum(encoded_test_sentences) == 4
    # four of them are OOV (is, multi, paradigm, language) ...
    assert encoded_test_sentences[0, 1] == 1
    # python occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == 1
    # a occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 1,
        "python": 1,
        "a": 1,
        "programming": 1,
    }


def test_sow_encoder_limit_vocab_and_top_words():
    # build a vocab of size 20 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = BowEncoder(skip_top_words=3, limit_vocabulary=20, mode="binary")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 20 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 20)
    # test sentence contains 7 words, but because OOV occurs 4 times and reduces to a binary value of 1, it sums up to 4
    assert np.sum(encoded_test_sentences) == 4
    # three of them are OOV (multi, paradigm, python, and a)
    assert encoded_test_sentences[0, 1] == 1
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == 1

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 1,
        "is": 1,
        "programming": 1,
        "language": 1,
    }


def test_sow_encoder_limit_vocab_and_top_words_oov_update_corner_cases():
    encoder = BowEncoder(skip_top_words=1, limit_vocabulary=60, mode="binary")
    encoder.prepare_vocabulary(corpus, show_progress=False)
    # here we test the tree cases, where the OOV is actually (or not) influenced by skip_top_words=1 (removal of and):
    #  - corpus[0] contains no OOV word(s) and does not contain 'and'
    #  - corpus[1] contains no OOV word(s) and also contains 'and'
    #  - corpus[4] contains OOV word(s) and also contains 'and'
    #  - test_sentence contains OOV word(s) but does not contain 'and'
    encoded_test_sentences = encoder.encode([corpus[0], corpus[1], corpus[4], test_sentence],
                                            show_progress=False)
    # no OOV + no 'and'
    assert encoded_test_sentences[0, 1] == 0
    # no OOV + 'and'
    assert encoded_test_sentences[1, 1] == 1
    # OOV + 'and'
    assert encoded_test_sentences[1, 1] == 1
    # OOV + no 'and'
    assert encoded_test_sentences[2, 1] == 1


def test_bow_encoder_default():
    encoder = BowEncoder(mode="count")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 100 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 100)
    # test sentence contains 7 words
    assert np.sum(encoded_test_sentences) == 7
    # two of them are OOV (multi and paradigm)
    assert encoded_test_sentences[0, 1] == 2
    # python occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == 1
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == 1
    # a occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == 1

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 2,
        "python": 1,
        "is": 1,
        "a": 1,
        "programming": 1,
        "language": 1,
    }


def test_bow_encoder_limit_vocab():
    # build a vocab of size 8 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)

    encoder = BowEncoder(limit_vocabulary=8, mode="count")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 8 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 8)
    # test sentence contains 7 words
    assert np.sum(encoded_test_sentences) == 7
    # four of them are OOV (is, multi, paradigm, language) ...
    assert encoded_test_sentences[0, 1] == 4
    # python occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == 1
    # a occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 4,
        "python": 1,
        "a": 1,
        "programming": 1,
    }


def test_bow_encoder_limit_vocab_and_top_words():
    # build a vocab of size 20 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = BowEncoder(skip_top_words=3, limit_vocabulary=20, mode="count")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 20 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 20)
    # test sentence contains 7 words
    assert np.sum(encoded_test_sentences) == 7
    # three of them are OOV (multi, paradigm, python, and a)
    assert encoded_test_sentences[0, 1] == 4
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == 1
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == 1
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == 1

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == {
        "<OOV>": 4,
        "is": 1,
        "programming": 1,
        "language": 1,
    }


def test_freq_encoder_default():
    encoder = BowEncoder(mode="freq")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 100 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 100)
    # test sentence has a relative size of 1 (all 7 words)
    assert np.sum(encoded_test_sentences) == approx(1, rel=1e-3)
    # two of them (overall 7) are OOV (multi and paradigm)
    assert encoded_test_sentences[0, 1] == approx(2/7., rel=1e-3)
    # python occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == approx(1/7., rel=1e-3)
    # is occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == approx(1/7., rel=1e-3)
    # a occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == approx(1/7., rel=1e-3)
    # programming occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1/7., rel=1e-3)
    # language occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == approx(1/7., rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": 2/7.,
        "python": 1/7.,
        "is": 1/7.,
        "a": 1/7.,
        "programming": 1/7.,
        "language": 1/7.,
    }, rel=1e-3)


def test_freq_encoder_limit_vocab():
    # build a vocab of size 20 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)

    encoder = BowEncoder(limit_vocabulary=8, mode="freq")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 8 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 8)
    # test sentence has a relative size of 1 (all 7 words)
    assert np.sum(encoded_test_sentences) == approx(1, rel=1e-3)
    # four of them are OOV (is, multi, paradigm, language) ...
    assert encoded_test_sentences[0, 1] == approx(4/7., rel=1e-3)
    # python occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == approx(1/7., rel=1e-3)
    # a occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == approx(1/7., rel=1e-3)
    # programming occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1/7., rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": 4/7.,
        "python": 1/7.,
        "a": 1/7.,
        "programming": 1/7.,
    }, rel=1e-3)


def test_freq_encoder_limit_vocab_and_top_words():
    # build a vocab of size 20 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = BowEncoder(skip_top_words=3, limit_vocabulary=20, mode="freq")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 20 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 20)
    # test sentence has a relative size of 1 (all 7 words)
    assert np.sum(encoded_test_sentences) == approx(1, rel=1e-3)
    # three of them are OOV (multi, paradigm, python, and a)
    assert encoded_test_sentences[0, 1] == approx(4/7., rel=1e-3)
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == approx(1/7., rel=1e-3)
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1/7., rel=1e-3)
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == approx(1/7., rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": 4/7.,
        "is": 1/7.,
        "programming": 1/7.,
        "language": 1/7.,
    }, rel=1e-3)


def test_tfidf_encoder_default():
    encoder = BowEncoder(mode="tfidf")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 100 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 100)
    # test sentence tfidf sum (over all 7 words)
    assert np.sum(encoded_test_sentences) == approx(9.706, rel=1e-3)
    # two of them are OOV (multi and paradigm)
    assert encoded_test_sentences[0, 1] == approx(3.898, rel=1e-3)
    # python occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == approx(0.826, rel=1e-3)
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == approx(1.386, rel=1e-3)
    # a occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == approx(1.029, rel=1e-3)
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1.178, rel=1e-3)
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == approx(1.386, rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": 3.898,
        "python": 0.826,
        "is": 1.386,
        "a": 1.029,
        "programming": 1.178,
        "language": 1.386,
    }, rel=1e-3)


def test_tfidf_encoder_limit_vocab():
    # build a vocab of size 8 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)

    encoder = BowEncoder(limit_vocabulary=8, mode="tfidf")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 8 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 8)
    # test sentence tfidf sum (over all 7 words)
    assert np.sum(encoded_test_sentences) == approx(8.529, rel=1e-3)
    # four of them are OOV (is, multi, paradigm, language) ...
    assert encoded_test_sentences[0, 1] == approx(5.494, rel=1e-3)
    # python occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("python")] == approx(0.826, rel=1e-3)
    # a occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("a")] == approx(1.029, rel=1e-3)
    # programming occurs once (out of 7 words)
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1.178, rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": 5.494,
        "python": 0.826,
        "a": 1.029,
        "programming": 1.178,
    }, rel=1e-3)


def test_tfidf_encoder_limit_vocab_and_top_words():
    # build a vocab of size 20 including:
    #  - reserved token <UNUSED> and <OOV>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = BowEncoder(skip_top_words=3, limit_vocabulary=20, mode="tfidf")
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence], show_progress=False)

    # vocabulary consists of overall 20 words and the test set contains only one text
    assert encoded_test_sentences.shape == (1, 20)
    # test sentence has a relative size of 1 (all 7 words)
    assert np.sum(encoded_test_sentences) == approx(9.706, rel=1e-3)
    # three of them are OOV (multi, paradigm, python, and a), however, current tfidf aggregation for oov is broken
    # TODO fix oov aggregation for top k (currently only implemented as: tfidf(OOV)+tfidf(top1)+tfidf(top2)+...)
    assert encoded_test_sentences[0, 1] > 3.898
    # is occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("is")] == approx(1.386, rel=1e-3)
    # programming occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("programming")] == approx(1.178, rel=1e-3)
    # language occurs once
    assert encoded_test_sentences[0, encoder.word_to_index("language")] == approx(1.386, rel=1e-3)

    # decode
    bow_dict = encoder.decode(encoded_test_sentences, 0, show_progress=False, ignore_zero_freq=True)

    assert bow_dict == approx({
        "<OOV>": encoded_test_sentences[0, 1],
        "is": 1.386,
        "programming": 1.178,
        "language": 1.386,
    }, rel=1e-3)


def test_sequence_encoder():
    encoder = TokenSequenceEncoder()
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False)

    # sentence consists of 7 words + <START> token
    assert encoded_test_sentences.shape == (2, 8)

    # first word is '<START>'
    assert encoded_test_sentences[0, 0] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 1] == 5
    # third word is 'is' (7th most common + 4 reserved token)
    assert encoded_test_sentences[0, 2] == 10
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 6
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 4] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 5] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common + 4 reserved token)
    assert encoded_test_sentences[0, 6] == 7
    # eighth word is 'language' (8th most common + 4 reserved token)
    assert encoded_test_sentences[0, 7] == 11

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :6],
        np.array([encoder.padding_token_index]*6))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 6] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 7] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["python", "is", "a", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["and"]

    # decode w/ control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<START>", "python", "is", "a", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "and"]


def test_sequence_encoder_limit_vocab():
    # build a vocab of size 10 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)
    encoder = TokenSequenceEncoder(limit_vocabulary=10)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False)

    # sentence consists of 7 words + <START> token
    assert encoded_test_sentences.shape == (2, 8)

    # first word is '<START>'
    assert encoded_test_sentences[0, 0] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 1] == 5
    # thord word is 'is' (not in the limited vocab -> OOV)
    assert encoded_test_sentences[0, 2] == encoder.oov_token_index
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 6
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 4] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 5] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common + 4 reserved token)
    assert encoded_test_sentences[0, 6] == 7
    # eighth word is 'language' (not in the limited vocab -> OOV)
    assert encoded_test_sentences[0, 7] == encoder.oov_token_index

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :6],
        np.array([encoder.padding_token_index]*6))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 6] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 7] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["python", "<OOV>", "a", "<OOV>", "<OOV>", "programming", "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["and"]

    # decode w/ control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<START>", "python", "<OOV>", "a", "<OOV>", "<OOV>", "programming", "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "and"]


def test_sequence_encoder_limit_vocab_and_top_words():
    # build a vocab of size 22 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = TokenSequenceEncoder(skip_top_words=3, limit_vocabulary=22)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "Python"], show_progress=False)

    # sentence consists of 7 words + <START> token
    assert encoded_test_sentences.shape == (2, 8)

    # first word is '<START>'
    assert encoded_test_sentences[0, 0] == encoder.start_token_index
    # second word is 'Python' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[0, 1] == encoder.oov_token_index
    # third word is 'is' (7th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 2] == 7
    # fourth word is 'a' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[0, 3] == encoder.oov_token_index
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 4] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 5] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 6] == 4
    # eighth word is 'language' (8th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 7] == 8

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :6],
        np.array([encoder.padding_token_index]*6))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 6] == encoder.start_token_index
    # second word is 'and' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[1, 7] == encoder.oov_token_index

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<OOV>", "is", "<OOV>", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["<OOV>"]

    # decode w/ control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<START>", "<OOV>", "is", "<OOV>", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "<OOV>"]


def test_truncated_sequence_encoder():
    encoder = TokenSequenceEncoder(default_length=5)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False)

    # padding to size 10 and two sentences
    assert encoded_test_sentences.shape == (2, 5)

    # first word is '<START>'
    assert encoded_test_sentences[0, 0] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 1] == 5
    # third word is 'is' (7th most common + 4 reserved token)
    assert encoded_test_sentences[0, 2] == 10
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 6
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 4] == encoder.oov_token_index

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :3],
        np.array([encoder.padding_token_index]*3))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 3] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 4] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<START>", "python", "is", "a", "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<START>", "and"]

    # decode w/o control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["python", "is", "a", "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["and"]

    # same same but with encoding specific length
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False, length=4)

    # padding to size 10 and two sentences
    assert encoded_test_sentences.shape == (2, 4)

    # first word is '<START>'
    assert encoded_test_sentences[0, 0] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 1] == 5
    # third word is 'is' (7th most common + 4 reserved token)
    assert encoded_test_sentences[0, 2] == 10
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 6

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :2],
        np.array([encoder.padding_token_index]*2))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 2] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 3] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<START>", "python", "is", "a"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<START>", "and"]

    # decode w/o control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["python", "is", "a"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["and"]


def test_padded_sequence_encoder():
    encoder = TokenSequenceEncoder(default_length=10)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False)

    # padding to size 10 and two sentences
    assert encoded_test_sentences.shape == (2, 10)

    # padding with '<PAD>' (2 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[0, :2],
        np.array([encoder.padding_token_index]*2))
    # first word is '<START>'
    assert encoded_test_sentences[0, 2] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 5
    # third word is 'is' (7th most common + 4 reserved token)
    assert encoded_test_sentences[0, 4] == 10
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 5] == 6
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 6] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 7] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common + 4 reserved token)
    assert encoded_test_sentences[0, 8] == 7
    # eighth word is 'language' (8th most common + 4 reserved token)
    assert encoded_test_sentences[0, 9] == 11

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :8],
        np.array([encoder.padding_token_index]*8))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 8] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 9] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<START>", "python", "is", "a", "<OOV>", "<OOV>", "programming",
                             "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "and"]

    # decode w/o control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["python", "is", "a", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["and"]


def test_padded_sequence_encoder_limit_vocab():
    # build a vocab of size 10 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)
    encoder = TokenSequenceEncoder(default_length=10, limit_vocabulary=10)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "and"], show_progress=False)

    # padding to size 10 and two sentences
    assert encoded_test_sentences.shape == (2, 10)

    # padding with '<PAD>' (2 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[0, :2],
        np.array([encoder.padding_token_index]*2))
    # first word is '<START>'
    assert encoded_test_sentences[0, 2] == encoder.start_token_index
    # second word is 'Python' (2nd most common + 4 reserved token)
    assert encoded_test_sentences[0, 3] == 5
    # thord word is 'is' (not in the limited vocab -> OOV)
    assert encoded_test_sentences[0, 4] == encoder.oov_token_index
    # fourth word is 'a' (3rd most common + 4 reserved token)
    assert encoded_test_sentences[0, 5] == 6
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 6] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 7] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common + 4 reserved token)
    assert encoded_test_sentences[0, 8] == 7
    # eighth word is 'language' (not in the limited vocab -> OOV)
    assert encoded_test_sentences[0, 9] == encoder.oov_token_index

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :8],
        np.array([encoder.padding_token_index]*8))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 8] == encoder.start_token_index
    # second word is 'and' (most common + 4 reserved token)
    assert encoded_test_sentences[1, 9] == 4

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<START>", "python", "<OOV>", "a", "<OOV>", "<OOV>", "programming",
                             "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "and"]

    # decode w/o control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["python", "<OOV>", "a", "<OOV>", "<OOV>", "programming", "<OOV>"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["and"]


def test_padded_sequence_encoder_limit_vocab_and_top_words():
    # build a vocab of size 22 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = TokenSequenceEncoder(default_length=10, skip_top_words=3, limit_vocabulary=22)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    # encode test sentence
    encoded_test_sentences = encoder.encode([test_sentence, "Python"], show_progress=False)

    # padding to size 10 and two sentences
    assert encoded_test_sentences.shape == (2, 10)

    # padding with '<PAD>' (2 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[0, :2],
        np.array([encoder.padding_token_index]*2))
    # first word is '<START>'
    assert encoded_test_sentences[0, 2] == encoder.start_token_index
    # second word is 'Python' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[0, 3] == encoder.oov_token_index
    # third word is 'is' (7th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 4] == 7
    # fourth word is 'a' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[0, 5] == encoder.oov_token_index
    # fifth word is 'multi' (unknown -> OOV)
    assert encoded_test_sentences[0, 6] == encoder.oov_token_index
    # sixth word is 'paradigm' (unknown -> OOV)
    assert encoded_test_sentences[0, 7] == encoder.oov_token_index
    # seventh word is 'programming' (4th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 8] == 4
    # eighth word is 'language' (8th most common - top-3 words + 4 reserved token)
    assert encoded_test_sentences[0, 9] == 8

    # padding with '<PAD>' (6 chars)
    np.testing.assert_array_equal(
        encoded_test_sentences[1, :8],
        np.array([encoder.padding_token_index]*8))
    # first word after is '<START>'
    assert encoded_test_sentences[1, 8] == encoder.start_token_index
    # second word is 'and' (not in the limited vocab (among top-3) -> OOV)
    assert encoded_test_sentences[1, 9] == encoder.oov_token_index

    # decode
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False)
    assert sequence_list == ["<PAD>", "<PAD>", "<START>", "<OOV>", "is", "<OOV>", "<OOV>", "<OOV>", "programming",
                             "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=True, show_padding=True)
    assert sequence_list == ["<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<START>", "<OOV>"]

    # decode w/o control chars
    sequence_list = encoder.decode(encoded_test_sentences, 0, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["<OOV>", "is", "<OOV>", "<OOV>", "<OOV>", "programming", "language"]
    sequence_list = encoder.decode(encoded_test_sentences, 1, show_progress=False, show_start=False, show_padding=False)
    assert sequence_list == ["<OOV>"]


class TestEmbeddingMatcher(AbstractEmbeddingMatcher):
    def __init__(self, encode_reserved_words):
        super().__init__(encode_reserved_words=encode_reserved_words)
        # some random unique embedding pattern
        self.vector_defs = {
            "python":      ["0.1",  "0.1",  "0.1",  "0.1"],
            "is":          ["-0.9", "-0.9", "-0.9", "-0.9"],
            "a":           ["0.2",  "0.3",  "0.4",  "0.5"],
            "multi":       ["-0.5", "-0.6", "-0.7", "-0.8"],
            "paradigm":    ["0.5",  "0.4",  "0.3",  "0.2"],
            "programming": ["-0.9", "-0.8", "-0.7", "-0.6"],
            "language":    ["0.4",  "0.6",  "0.6",  "0.4"],
        }

    def get_vector_source(self):
        vectors = list(WordVector(word, vec) for word, vec in self.vector_defs.items())

        return len(self.vector_defs), 4, vectors


def test_embedding_matcher():
    encoder = TokenSequenceEncoder(default_length=10)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    #
    # do not encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=False)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 102 words/token with 4 dimensions each
    expected = np.zeros((102, 4))
    # every word/token has an embedded representation
    # 'python' is 2nd most common + 4 reserved token -> index:5
    expected[5] = np.array(matcher.vector_defs["python"])
    # 'is' is 7th most common + 4 reserved token -> index:10
    expected[10] = np.array(matcher.vector_defs["is"])
    # 'a' is 3rd most common + 4 reserved token -> index:6
    expected[6] = np.array(matcher.vector_defs["a"])
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common + 4 reserved token -> index:7
    expected[7] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common + 4 reserved token -> index:11
    expected[11] = np.array(matcher.vector_defs["language"])

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )

    #
    # do encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=True)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 102 words/token with 4 dimensions each (filled with OOV --> one vector (normalized by 1e-16))
    expected = np.ones((102, 4)) * matcher.normalize_reserved_embeddings_by

    # we only need to update embeddings not equal to OOV
    # reserved words embeddings:
    # <PAD> -> zero vector
    expected[0] = np.zeros(4)
    # <OOV> -> one vector (normalized by 1e-16) ..actually not necessary
    expected[1] = np.array([1]*4) * matcher.normalize_reserved_embeddings_by
    # <START> -> minus one vector (normalized by 1e-16)
    expected[2] = np.array([-1]*4) * matcher.normalize_reserved_embeddings_by
    # <END> -> alternating(one, minus-one) vector (normalized by 1e-16)
    expected[3] = np.array([1, -1]*2) * matcher.normalize_reserved_embeddings_by

    # 'python' is 2nd most common + 4 reserved token -> index:5
    expected[5] = np.array(matcher.vector_defs["python"])
    # 'is' is 7th most common + 4 reserved token -> index:10
    expected[10] = np.array(matcher.vector_defs["is"])
    # 'a' is 3rd most common + 4 reserved token -> index:6
    expected[6] = np.array(matcher.vector_defs["a"])
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common + 4 reserved token -> index:7
    expected[7] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common + 4 reserved token -> index:11
    expected[11] = np.array(matcher.vector_defs["language"])

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )


def test_embedding_matcher_limit_vocab():
    # build a vocab of size 10 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus: and(8), python(7), a(4), programming(3), has(3), and the(3)
    encoder = TokenSequenceEncoder(default_length=10, limit_vocabulary=10)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    #
    # do not encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=False)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 102 words/token with 4 dimensions each
    expected = np.zeros((10, 4))
    # only few words have an embedded representation
    # 'python' is 2nd most common + 4 reserved token -> index:5
    expected[5] = np.array(matcher.vector_defs["python"])
    # 'is' is 7th most common + 4 reserved token -> OOV
    # 'a' is 3rd most common + 4 reserved token -> index:6
    expected[6] = np.array(matcher.vector_defs["a"])
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common + 4 reserved token -> index:7
    expected[7] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common + 4 reserved token -> OOV

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )

    #
    # do encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=True)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 102 words/token with 4 dimensions each (filled with OOV --> one vector (normalized by 1e-16))
    expected = np.ones((10, 4)) * matcher.normalize_reserved_embeddings_by

    # we only need to update embeddings not equal to OOV
    # reserved words embeddings:
    # <PAD> -> zero vector
    expected[0] = np.zeros(4)
    # <OOV> -> one vector (normalized by 1e-16) ..actually not necessary
    expected[1] = np.array([1]*4) * matcher.normalize_reserved_embeddings_by
    # <START> -> minus one vector (normalized by 1e-16)
    expected[2] = np.array([-1]*4) * matcher.normalize_reserved_embeddings_by
    # <END> -> alternating(one, minus-one) vector (normalized by 1e-16)
    expected[3] = np.array([1, -1]*2) * matcher.normalize_reserved_embeddings_by

    # only few words have an embedded representation
    # 'python' is 2nd most common + 4 reserved token -> index:5
    expected[5] = np.array(matcher.vector_defs["python"])
    # 'is' is 7th most common + 4 reserved token -> OOV
    # 'a' is 3rd most common + 4 reserved token -> index:6
    expected[6] = np.array(matcher.vector_defs["a"])
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common + 4 reserved token -> index:7
    expected[7] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common + 4 reserved token -> OOV

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )


def test_embedding_matcher_limit_vocab_and_top_words():

    # build a vocab of size 22 including:
    #  - reserved token <PAD>, <OOV>, <START>, and <END>
    #  - plus the top 6 words in the corpus:
    #      programming(3), has(3), the(3), is(2), language(2), by(2), van(2), rossum(2), in(2), that(2), it(2),
    #      large(2), community(2), as(2), are(2), cpython(2), of(2), software(2)
    #  - ignored words (top 3): and(8), python(7), a(4)

    encoder = TokenSequenceEncoder(default_length=10, skip_top_words=3, limit_vocabulary=22)
    encoder.prepare_vocabulary(corpus, show_progress=False)

    #
    # do not encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=False)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 22 words/token with 4 dimensions each
    expected = np.zeros((22, 4))
    # only few words have an embedded representation
    # 'python' is 2nd most common -> OOV
    # 'is' is 7th most common - top-3 words + 4 reserved token -> index:10
    expected[7] = np.array(matcher.vector_defs["is"])
    # 'a' is 3rd most common -> OOV
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common - top-3 words + 4 reserved token -> index:7
    expected[4] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common - top-3 words + 4 reserved token -> index:11
    expected[8] = np.array(matcher.vector_defs["language"])

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )

    #
    # do encode reserved words
    #
    matcher = TestEmbeddingMatcher(encode_reserved_words=True)

    with raises(ValueError) as e_info:
        _ = matcher.embedding_matrix

    matcher.reload_embeddings(token_encoder=encoder, show_progress=False)

    # expect matrix for 102 words/token with 4 dimensions each (filled with OOV --> one vector (normalized by 1e-16))
    expected = np.ones((22, 4)) * matcher.normalize_reserved_embeddings_by

    # we only need to update embeddings not equal to OOV
    # reserved words embeddings:
    # <PAD> -> zero vector
    expected[0] = np.zeros(4)
    # <OOV> -> one vector (normalized by 1e-16) ..actually not necessary
    expected[1] = np.array([1]*4) * matcher.normalize_reserved_embeddings_by
    # <START> -> minus one vector (normalized by 1e-16)
    expected[2] = np.array([-1]*4) * matcher.normalize_reserved_embeddings_by
    # <END> -> alternating(one, minus-one) vector (normalized by 1e-16)
    expected[3] = np.array([1, -1]*2) * matcher.normalize_reserved_embeddings_by

    # 'python' is 2nd most common -> OOV
    # 'is' is 7th most common - top-3 words + 4 reserved token -> index:10
    expected[7] = np.array(matcher.vector_defs["is"])
    # 'a' is 3rd most common -> OOV
    # 'multi' is unknown -> OOV
    # 'paradigm' is unknown -> OOV
    # 'programming' is 4th most common - top-3 words + 4 reserved token -> index:7
    expected[4] = np.array(matcher.vector_defs["programming"])
    # 'language' is 8th most common - top-3 words + 4 reserved token -> index:11
    expected[8] = np.array(matcher.vector_defs["language"])

    np.testing.assert_array_equal(
        matcher.embedding_matrix,
        expected
    )
