class InvalidPathError(Exception):
    """Raised when the output path exists."""
    pass


class AbsolutePathError(Exception):
    """Raised when the output path is absolute."""
    pass


class InvalidTileSizeError(Exception):
    """Raised when the tile size is invalid."""
    pass


class InsuffientDataError(Exception):
    """Raised when more data than available is requested."""
    pass


class OutputPathExistsError(Exception):
    """Raised when the output path exists."""
    pass
