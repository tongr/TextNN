from __future__ import unicode_literals
from distutils import dir_util
from pytest import fixture
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

    assert ["1", "2	2", "", "	4"] == list(read_text_file_lines(test_data_file, gzip=True))

    assert ["", "	4"] == list(read_text_file_lines(test_data_file, ignore_first_n_lines=2, gzip=True))


def test_read_text_file(datadir):
    test_data_file = datadir / "test.file"

    assert "1\n2\t2\n\n\t4" == read_text_file(test_data_file)


def test_write_text_file(datadir):
    test_data_file = datadir / "new_test.file"
    test_text = "this is a test\ntext"

    write_text_file(test_text, file_path=test_data_file)
    assert test_text == read_text_file(test_data_file)

