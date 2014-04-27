class Wrapper(object):

    """Construct an anonymous object with specific fields.

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
