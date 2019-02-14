from typing import Iterable

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
