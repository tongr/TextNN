from pytest import raises

from textnn.utils.encoding.label import *


def test_label_encoder_init_labeled_data():
    encoder = LabelEncoder(labeled_data=[
        ("test text 1", 1),
        ("test text 2", 0),
        ("test text 3", 1),
        ("test text 4", 3),
    ])

    assert encoder.num_classes == 4


def test_label_encoder_init_label_sample():
    encoder = LabelEncoder(label_sample_it=[1, 5, 1, 0])

    assert encoder.num_classes == 6


def test_label_encoder_make_categorical_labels():
    encoder = LabelEncoder(num_classes=3)

    assert encoder.num_classes == 3

    np.testing.assert_array_equal(
        encoder.make_categorical(y_labels=[
            0,
            0,
            1,
            0,
            2,
            1,
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ])
    )

    # make sure the num_classes restrict labels > 2
    with raises(Exception) as e_info:
        _ = encoder.make_categorical([3])


def test_label_encoder_make_categorical_labeled_data():
    encoder = LabelEncoder(num_classes=3)

    assert encoder.num_classes == 3

    np.testing.assert_array_equal(
        encoder.make_categorical(labeled_data=[
            ("text 1", 1),
            ("text 2", 0),
            ("text 3", 1),
            ("text 4", 2),
        ]),
        np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
    )

    # make sure the num_classes restrict labels > 2
    with raises(Exception) as e_info:
        _ = encoder.make_categorical([3])


def test_label_encoder_max_category():
    encoder = LabelEncoder(num_classes=3)

    assert encoder.num_classes == 3

    # clear decisions
    np.testing.assert_array_equal(
        encoder.max_category(np.array([
            [0,  1,  0],
            [10, 1,  1],
            [0,  .3, .1],
            [.1, .1, .2],
        ])),
        np.array([
            1,
            0,
            1,
            2,
        ])
    )

    # unclear decisions (greedy take first)
    np.testing.assert_array_equal(
        encoder.max_category(np.array([
            [0,  1,  1],
            [.1, .1, .1],
            [.1, .2, .2],
        ])),
        np.array([
            1,
            0,
            1,
        ])
    )

    # make sure the num_classes restrict labels > 2
    with raises(Exception) as e_info:
        _ = encoder.max_category(np.array([[0, 0, 0, 1]]))


