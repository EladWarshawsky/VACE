# Minimal EasyDict fallback to avoid external dependency
# Provides attribute-style access to dict items.

class EasyDict(dict):
    """A minimal EasyDict implementation.

    Example:
        d = EasyDict({'a': 1})
        assert d.a == 1
        d.b = 2
        assert d['b'] == 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        # allow normal attribute assignment for internal attributes
        if name.startswith('_') or name in ('__setstate__',):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    # ensure nested dicts become EasyDicts on update
    def update(self, *args, **kwargs):
        def _convert(v):
            if isinstance(v, dict) and not isinstance(v, EasyDict):
                return EasyDict(v)
            return v
        other = {}
        if args:
            if len(args) != 1:
                raise TypeError('update expected at most 1 arguments, got %d' % len(args))
            other = dict(args[0])
        other.update(kwargs)
        for k, v in other.items():
            super().__setitem__(k, _convert(v))
