import io
import sys
from textnn.utils import ProgressIterator


#inspired by https://stackoverflow.com/a/34738440
def capture_sysout(cmd):
    capturedOutput = io.StringIO()                  # Create StringIO object
    sys.stdout = capturedOutput                     #  and redirect stdout.
    cmd()                                     # Call function.
    sys.stdout = sys.__stdout__                     # Reset redirect.
    return capturedOutput.getvalue()                # Now works as before.


def test_progress_iterator():
    def progress_generator():
        sum(ProgressIterator([1, 2, 3], interval=0, description=""))

    report = capture_sysout(cmd=progress_generator)
    lines = report.strip().split("\n")

    # expected result (with changing numbers):
    # 1/3 [=========>....................] - ETA: 7s
    # 2/3 [===================>..........] - ETA: 1s
    # 3/3 [==============================] - 4s 1s/step
    assert lines[0].startswith("1/3")
    assert "ETA: " in lines[0]
    assert lines[1].startswith("2/3")
    assert "ETA: " in lines[1]
    assert lines[2].startswith("3/3")
    assert lines[2].endswith("s/step")


def test_progress_iterator_with_statement():
    def progress_generator():
        with ProgressIterator([1,2,3], interval=0, description="") as it:
            sum(it)

    report = capture_sysout(cmd=progress_generator)
    lines = report.strip().split("\n")

    # expected result (with changing numbers):
    # 1/3 [=========>....................] - ETA: 7s
    # 2/3 [===================>..........] - ETA: 1s
    # 3/3 [==============================] - 4s 1s/step
    assert lines[0].startswith("1/3")
    assert "ETA: " in lines[0]
    assert lines[1].startswith("2/3")
    assert "ETA: " in lines[1]
    assert lines[2].startswith("3/3")
    assert lines[2].endswith("s/step")

