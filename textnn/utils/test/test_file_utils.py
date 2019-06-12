from __future__ import unicode_literals
from distutils import dir_util
from pytest import fixture, raises
import os
from pathlib import Path

from textnn.utils import join_name, read_text_file_lines, read_text_file, write_text_file

# inspired by https://stackoverflow.com/a/29631801
@fixture
def datadir(tmpdir, request) -> Path:
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return Path(tmpdir)


def test_read_text_file_lines(datadir):
    test_data_file = datadir / "test.file"

    assert ["1", "2	2", "", "	4"] == list(read_text_file_lines(test_data_file))

    assert ["", "	4"] == list(read_text_file_lines(test_data_file, ignore_first_n_lines=2))


def test_read_text_file_lines_gzip(datadir):
    test_data_file = datadir / "test.file.gz"

    assert ["1", "2	2", "", "	4"] == list(read_text_file_lines(test_data_file, compression="gzip"))

    assert ["", "	4"] == list(read_text_file_lines(test_data_file, ignore_first_n_lines=2, compression="gzip"))


def test_read_text_file_lines_bz2(datadir):
    test_data_file = datadir / "test.file.bz2"

    assert ["1", "2	2", "", "	4"] == list(read_text_file_lines(test_data_file, compression="bz2"))

    assert ["", "	4"] == list(read_text_file_lines(test_data_file, ignore_first_n_lines=2, compression="bz2"))


def test_read_text_file_lines_unknown_compression(datadir):
    test_data_file = datadir / "test.file"

    # make sure unknown compression types throw exceptions
    with raises(ValueError) as e_info:
        read_text_file_lines(test_data_file, compression="???")


def test_read_text_file_lines_unknown_file(datadir):
    test_data_file = datadir / "test.file.???"

    # make sure unknown files throw exceptions
    with raises(AssertionError) as e_info:
        read_text_file_lines(test_data_file)


def test_read_text_file(datadir):
    test_data_file = datadir / "test.file"

    assert "1\n2\t2\n\n\t4" == read_text_file(test_data_file)


def test_write_text_file(datadir):
    test_data_file = datadir / "new_test.file"
    test_text = "this is a test\ntext"

    write_text_file(test_text, file_path=test_data_file)
    assert test_text == read_text_file(test_data_file)


def test_join_name():
    assert "test__file" == join_name(["test", "file"])
    assert "test__file" == join_name(["test", None, "file", None], ignore_none=True)
    assert "test<>file" == join_name(["test", "file"], separator="<>")
