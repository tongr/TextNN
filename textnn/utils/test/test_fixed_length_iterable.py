from pytest import raises
from textnn.utils import FixedLengthIterable


def test_fixed_length_iterable_lazy():
    data = [1, 2, 3, 4, 5]

    iterable = FixedLengthIterable(iterable=data)

    # make sure the lazy form does not provide iterable length (yet)
    with raises(TypeError) as e_info:
        _ = len(iterable)

    assert sum(iterable) == 15

    # once consumed, the length should be available
    assert len(iterable) == 5

    assert sum(iterable) == 15


def test_fixed_length_iterable_greedy():
    data = [1, 2, 3, 4, 5]

    iterable = FixedLengthIterable(iterable=data, lazy=False)

    # while accessing len(), we will iterate the source the first time
    assert len(iterable) == 5

    assert sum(iterable) == 15


def test_fixed_length_iterable_known_length():
    data = [1, 2, 3, 4, 5]

    iterable = FixedLengthIterable(iterable=data, length=3)

    # the length is predefined
    assert len(iterable) == 3

    assert sum(iterable) == 15

    # even though we just iterated over 5 elements, the length stays at the predefined value
    assert len(iterable) == 3


def test_fixed_length_generator_source_lazy():
    def gen_source():
        return (x for x in [1, 2, 3, 4, 5])

    iterable = FixedLengthIterable(gen_source=gen_source)

    # make sure the lazy form does not provide iterable length (yet)
    with raises(TypeError) as e_info:
        _ = len(iterable)

    assert sum(iterable) == 15

    # once consumed, the length should be available
    assert len(iterable) == 5

    assert sum(iterable) == 15


def test_fixed_length_generator_source_greedy():
    def gen_source():
        return (x for x in [1, 2, 3, 4, 5])

    iterable = FixedLengthIterable(gen_source=gen_source, lazy=False)

    # while accessing len(), we will iterate the source the first time
    assert len(iterable) == 5

    assert sum(iterable) == 15
