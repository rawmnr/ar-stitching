import contextlib

def raises(exc):
    @contextlib.contextmanager
    def _():
        try:
            yield
        except exc as e:
            pass
        else:
            raise AssertionError(f"{exc} not raised")
    return _()
