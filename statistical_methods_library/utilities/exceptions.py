class SMLError(Exception):
    """
    Error raised by the SML.
    """

    pass


class ValidationError(SMLError):
    """
    Error raised when validating the input data frame.
    """

    pass
