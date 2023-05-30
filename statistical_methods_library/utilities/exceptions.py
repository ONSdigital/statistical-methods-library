"""
Exception classes used by this library.

For Copyright information, please see LICENCE.
"""


class SMLError(Exception):
    """
    Base class for errors raised by statistical methods.
    """

    pass


class ValidationError(SMLError):
    """
    Error raised when validating the input data frame.
    """

    pass
