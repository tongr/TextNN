import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def join_name(name_parts: list, separator: str = "__", ignore_none: bool = True) -> str:
    """
    joins individual parts of a name (i.e., file name consisting of different elements)
    :param name_parts: the elements to join
    :param separator: the separator to be used to join
    :param ignore_none: if True, None values will not be shown in the joined string
    :return: the joined string representation of the `name_parts`
    """
    # create name by joining all of the following elements with a `separator` (maybe: remove empty strings / None)
    return separator.join(e for e in name_parts if not ignore_none or e is not None)


def plot2file(file: Path, x_values: list, y_series: Dict[str, list],
              title: str = None, x_label: str = None, y_label: str = None, series_styles: List[str] = None,
              shaded_areas: Iterable[Tuple[str, str, str, float]] = None,
              ) -> None:
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
    :param shaded_areas: definition of the shaded areas to plot, where each area is defined by a tuple consisting of:
    1. name of first y-series<br/>
    2. name of second y-series<br/>
    3. color of the shaded area (e.g., k)<br/>
    4. alpha-value of the shaded area
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

        max_len = max(len(l) for l in [x_values]+list(y_series.values()))
        x_values = x_values + [None] * (max_len - len(x_values))
        for idx, (series_label, y_values) in enumerate(y_series.items()):
            style = series_styles[idx % len(series_styles)]
            y_values = y_values + [None] * (max_len - len(y_values))
            plt.plot(x_values, y_values, style, label=series_label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        if shaded_areas:
            for area_y1_name, area_y2_name, area_color, area_alpha in shaded_areas:
                area_y1 = y_series[area_y1_name] + [None] * (max_len - len(y_series[area_y1_name]))
                area_y2 = y_series[area_y2_name] + [None] * (max_len - len(y_series[area_y2_name]))
                plt.fill_between(x_values, area_y1, area_y2, color=area_color, alpha=area_alpha)

        if not file.parent.exists():
            file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(file))
        plt.clf()
    except ImportError:
        logging.warning(f"Unable to create plot in {file}, missing Matplotlib dependency!")


def read_text_file(file_path: Path) -> str:
    """
    read text file contents
    :param file_path: the file path to read
    :return: the contents of the file
    """
    with open(str(file_path), 'r', encoding='utf8') as file:
        text = file.read()
    return text


def write_text_file(text: str, file_path: Path):
    """
    write text to file
    :param text: the text to persist
    :param file_path: the file path to use
    """
    with open(str(file_path), 'w', encoding='utf8') as file:
        file.write(text)
