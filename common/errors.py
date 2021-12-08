class AbsolutePathError(Exception):
    """Raised when the output path is absolute."""
    pass


class InsuffientDataError(Exception):
    """Raised when more data than available is requested."""
    pass


class InvalidPathError(Exception):
    """ Raised when an invalid path is given """
    pass


class InvalidTileSizeError(Exception):
    """Raised when the tile size is invalid."""
    pass


class LegacyModeError(Exception):
    """ Raised when legacy modes is used on incompatible data """
    pass


class OutputPathExistsError(Exception):
    """Raised when the output path exists."""
    pass
