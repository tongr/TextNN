from pytest import raises

from textnn.utils import skip


def test_skip_start():
    data = [1, 2, 3, 4, 5]

    assert list(skip(data, at_start=0)) == [1, 2, 3, 4, 5]
    assert list(skip(data, at_start=1)) == [2, 3, 4, 5]
    assert list(skip(data, at_start=3)) == [4, 5]
    assert list(skip(data, at_start=5)) == []
    assert list(skip(data, at_start=7)) == []


def test_skip_start_neg():
    data = [1, 2, 3, 4, 5]

    assert list(skip(data, at_start=-1)) == [1]
    assert list(skip(data, at_start=-3)) == [1, 2, 3]
    assert list(skip(data, at_start=-5)) == [1, 2, 3, 4, 5]
    assert list(skip(data, at_start=-7)) == [1, 2, 3, 4, 5]


def test_skip_end():
    data = [1, 2, 3, 4, 5]

    assert list(skip(data, at_end=0)) == [1, 2, 3, 4, 5]
    assert list(skip(data, at_end=1)) == [1, 2, 3, 4]
    assert list(skip(data, at_end=3)) == [1, 2]
    assert list(skip(data, at_end=5)) == []
    assert list(skip(data, at_end=7)) == []


def test_skip_end_neg():
    data = [1, 2, 3, 4, 5]

    # make sure the negative end offsets are not allowed (yet)
    with raises(Exception) as e_info:
        _ = skip(data, at_end=-1)


def test_skip_start_and_end():
    data = [1, 2, 3, 4, 5, 6, 7]

    assert list(skip(data, at_start=0, at_end=0)) == [1, 2, 3, 4, 5, 6, 7]
    assert list(skip(data, at_start=1, at_end=1)) == [2, 3, 4, 5, 6]
    assert list(skip(data, at_start=2, at_end=2)) == [3, 4, 5]
    assert list(skip(data, at_start=2, at_end=3)) == [3, 4]
    assert list(skip(data, at_start=3, at_end=2)) == [4, 5]
    assert list(skip(data, at_start=3, at_end=4)) == []
    assert list(skip(data, at_start=4, at_end=4)) == []


def test_skip_start_neg_and_end():
    data = [1, 2, 3, 4, 5]

    assert list(skip(data, at_start=-1, at_end=1)) == []
    assert list(skip(data, at_start=-2, at_end=1)) == [1]
    assert list(skip(data, at_start=-2, at_end=2)) == []
    assert list(skip(data, at_start=-2, at_end=3)) == []
    assert list(skip(data, at_start=-3, at_end=1)) == [1, 2]
    assert list(skip(data, at_start=-3, at_end=2)) == [1]
