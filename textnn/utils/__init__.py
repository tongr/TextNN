import logging
from pathlib import Path
from typing import Dict, Iterable, List

from keras.utils.generic_utils import Progbar, time


class ProgressIterator(Progbar):
    """
    Enables progress logging while iterating over collections such as lists and iterables.
    """

    def __init__(self, iterable: Iterable, description: str = None, target: int = None, interval: int = 1):
        """
        Create a new progress logging iterator instance
        :param iterable: the base iterable to iterate while logging progress
        :param target: the target (maximum) number of iterations to expect
        :param interval: the minimum interval between progress feedback (in seconds)
        """
        self._iterator = iter(iterable)
        self._progress = 0

        if target is None:
            try:
                # noinspection PyTypeChecker
                target = len(iterable)
            except TypeError:
                target = None
        # TODO check further parameter: width = 30, verbose = 1, stateful_metrics = None
        super().__init__(target=target, interval=interval)
        print(description if description is not None else "Processing ...")
        self._last_update = self._start

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = next(self._iterator)
            self.inc()
            return value
        except StopIteration:
            self.finish()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def __enter__(self):
        return self

    def inc(self):
        self._progress += 1

        if time.time() - self._last_update >= self.interval:
            self._seen_so_far = self._progress
            self.update(self._progress)

    def finish(self):
        # All done. Finalize progressbar.
        if self._progress != self._seen_so_far:
            # Force update by "hiding" last update time.
            self._last_update = 0
            self._seen_so_far = self._progress
            self.update(self.target if self.target is not None else self._progress)


def plot2file(file: Path, x_values: list, y_series: Dict[str, list],
              title: str = None, x_label: str = None, y_label: str = None, series_styles: List[str] = None):
    """
    plot the given data to a file
    :param file: storage location of the plot
    :param x_values: list of x values to plot
    :param y_series: dictionary of named(key) lists containing the y values to plot
    :param title: the plot title
    :param x_label: the x axis label
    :param y_label: the y axis label
    :param series_styles: the styles of the `y_series` (see "Format Strings" in
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)
    """
    try:
        import matplotlib.pyplot as plt
        # inspired by the tensorflow introduction of François Chollet
        # https://www.tensorflow.org/tutorials/keras/basic_text_classification
        if not series_styles:
            # default styles: iterate through different combinations
            series_styles = list(f"{color}{marker}{line}"
                                for line in "- -- -. :".split()
                                for marker in " o v ^ s P X".split(" ")
                                for color in "b r g c m y k".split())

        for idx, (series_label, y_values) in enumerate(y_series.items()):
            style = series_styles[idx % len(series_styles)]
            plt.plot(x_values, y_values, style, label=series_label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        if not file.parent.exists():
            file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(file))
        plt.clf()
    except ImportError:
        logging.warning(f"Unable to create plot in {file}, missing Matplotlib dependency!")
