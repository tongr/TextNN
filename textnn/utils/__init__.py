try:
    import ujson as json
except ImportError:
    import json as json
import itertools
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Union, Generator, Optional, Callable

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
            target = FixedLengthIterable.try_get_len(iterable)
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


class FixedLengthIterable(Iterable):
    """
    Enables len() attributes for Iterables and Generator sources to facilitate progre
    """
    def __init__(self, iterable: Optional[Iterable] = None,
                 gen_source: Optional[Callable[[], Generator]] = None,
                 length: Optional[int] = None,
                 lazy: bool = True):
        self._iterable = iterable
        self._gen_source = gen_source
        self.length = length
        self._lazy = lazy

    def __len__(self):
        if self.length is None:
            if self._lazy:
                raise TypeError("object has unknown len()")
            self.length = sum(1 for _ in self)
        return self.length

    @staticmethod
    def try_get_len(iterable: Iterable) -> Optional[int]:
        try:
            # noinspection PyTypeChecker
            return len(iterable)
        except TypeError:
            return None

    def _new_iter(self):
        if self._iterable is not None:
            return iter(self._iterable)
        return self._gen_source()

    def __iter__(self) -> Iterator:
        # if the length is unknown, store it while iterating and update the internal state
        if self.length is None:
            iterator = self._new_iter()

            def counting_gen():
                count = 0
                for x in iterator:
                    count += 1
                    yield x
                self.length = count
            return counting_gen()

        return self._new_iter()


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


def plot_to_file(file: Path,
                 x_values: list,
                 y_series: List[tuple],
                 title: str = None, x_label: str = None, y_label: str = None, default_series_styles: List[str] = None,
                 shaded_areas: Iterable[Tuple[Union[str, list], Union[str, list], str, float]] = None,
                 ) -> None:
    """
    plot the given data to a file
    :param file: storage location of the plot
    :param x_values: list of x values to plot
    :param y_series: list of y-series to plot, where each seriesis defined by a tuple consisting of:<br/>
    1. name/label of first y-series<br/>
    2. y-values of all data points to plot (order according to `x_values`)<br/>
    3. Optional: style definition of the series (see "Format Strings" in
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)<br/>
    4. Optional: if False the series will not be shown in the legend (default: True)
    :param title: the plot title
    :param x_label: the x axis label
    :param y_label: the y axis label
    :param default_series_styles: the styles of the `y_series` if not given in the tuple (see "Format Strings" in
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)
    :param shaded_areas: list of shaded area definitions to plot, where each area is defined by a tuple consisting
    of:<br/>
    1. name of first y-series or list of y-values of the first series<br/>
    2. name of second y-series or list of y-values of the second series<br/>
    3. color of the shaded area (e.g., k)<br/>
    4. alpha-value of the shaded area
    """
    try:
        import matplotlib.pyplot as plt
        # inspired by the tensorflow introduction of François Chollet
        # https://www.tensorflow.org/tutorials/keras/basic_text_classification
        if not default_series_styles:
            # default styles: iterate through different combinations
            default_series_styles = list(f"{color}{marker}{line}"
                                         for line in "- -- -. :".split()
                                         for marker in " o v ^ s P X".split(" ")
                                         for color in "b r g c m y k".split())

        legend_handles = []
        legend_labels = []
        cached_y_series_data = {}
        max_len = max([len(x_values)] + list(len(s[1]) for s in y_series))
        x_values = x_values + [None] * (max_len - len(x_values))
        for idx, data_tuple in enumerate(y_series):
            # element 0: label
            series_label = data_tuple[0]
            # element 1: y-values
            y_values = data_tuple[1] + [None] * (max_len - len(data_tuple[1]))
            cached_y_series_data[series_label] = y_values
            # element 2 (optional): style
            if len(data_tuple) > 2:
                style = data_tuple[2]
            else:
                style = default_series_styles[idx % len(default_series_styles)]
            series_handle, = plt.plot(x_values, y_values, style, label=series_label)
            # element 3 (optional): show in legend (default: True)
            if len(data_tuple) < 4 or data_tuple[3]:
                # add series to legend
                legend_handles.append(series_handle)
                legend_labels.append(series_label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if legend_handles:
            plt.legend(handles=legend_handles, labels=legend_labels)

        if shaded_areas:
            for area_y1, area_y2, area_color, area_alpha in shaded_areas:
                y1_series: list = cached_y_series_data[area_y1] if not isinstance(area_y1, list) else \
                    area_y1 + [None] * (max_len - len(area_y1))
                y2_series: list = cached_y_series_data[area_y2] if not isinstance(area_y2, list) else \
                    area_y2 + [None] * (max_len - len(area_y2))

                plt.fill_between(x=x_values,
                                 y1=y1_series,
                                 y2=y2_series,
                                 color=area_color,
                                 alpha=area_alpha)

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
    assert file_path.exists(), f"Unable to find text file {file_path}!"
    with open(str(file_path), 'r', encoding='utf8') as file:
        text = file.read()
    return text


def read_text_file_lines(file_path: Path, ignore_first_n_lines: int = 0, compression: str = None) -> Iterable[str]:
    """
    read lines text file contents
    :param file_path: the file path to read
    :param ignore_first_n_lines: the number of lines to be ignored at the beginning of the file
    :param compression: if None, the reader does not expect any compression, further options: `"gzip"/"gz"` and `"bz2"`
    :return: the line ontents of the file
    """
    assert file_path.exists(), f"Unable to find text file {file_path}!"
    if "gz" == compression or "gzip" == compression:
        def line_source() -> Generator[str, None, None]:
            import gzip
            with gzip.open(str(file_path), 'rt', encoding='utf8') as file:
                for idx, line in enumerate(file):
                    if not idx < ignore_first_n_lines:
                        yield line
    elif "bz2" == compression or "bzip2" == compression:
        def line_source() -> Generator[str, None, None]:
            import bz2
            with bz2.open(str(file_path), 'rt', encoding='utf8') as file:
                for idx, line in enumerate(file):
                    if not idx < ignore_first_n_lines:
                        yield line
    elif compression is None:
        def line_source() -> Generator[str, None, None]:
            with open(str(file_path), 'r', encoding='utf8') as file:
                for idx, line in enumerate(file):
                    if not idx < ignore_first_n_lines:
                        yield line
    else:
        raise ValueError(f"Unknown compression type '{compression}'!")

    # remove EOL
    return map(lambda line: line.rstrip("\n"), line_source())


def write_text_file(text: str, file_path: Path):
    """
    write text to file
    :param text: the text to persist
    :param file_path: the file path to use
    """
    assert file_path.parent.exists(), f"Unable to find folder {file_path} for output text file!"
    with open(str(file_path), 'w', encoding='utf8') as file:
        file.write(text)


# inspired by the solution of Abhijit: https://stackoverflow.com/a/18836614
def skip(iterable, at_start=0, at_end=0) -> Union[Iterator, Generator]:
    """
    Enables to skip specific parts at the start or end of an iterable.
    <b>Please Note!</b> Do not use this method for lists, use the builtin slicing functions instead.
    :param iterable: the iterable to truncate
    :param at_start: if positive (`n`), the first `n` elements are ignored in the resulting iterable, if negative (`n`),
    only the first `n` elements from the iterable will be returned
    :param at_end: if positive (`n`), the last `n` elements are ignored in the resulting iterable, negative values are
    not yet supported
    :return: the truncated iterator
    """
    it = iter(iterable)
    if at_start > 0:
        it = itertools.islice(it, at_start, None)
    elif at_start < 0:
        it = (v for idx, v in enumerate(it) if idx < -at_start)

    if at_end > 0:
        it, it1 = itertools.tee(it)
        it1 = itertools.islice(it1, at_end, None)
        return (next(it) for _ in it1)
    elif at_end < 0:
        # FIXME implement it
        raise NotImplementedError

    return it
