"""Custom exceptions for FL platform."""


class FLError(Exception):
    """Base exception for all FL platform errors."""

    pass


class ConfigurationError(FLError):
    """Raised when configuration is invalid."""

    pass


class DataLoadingError(FLError):
    """Raised when data loading fails."""

    pass


class ModelError(FLError):
    """Raised when model operations fail."""

    pass


class AggregationError(FLError):
    """Raised when aggregation fails."""

    pass


class PrivacyError(FLError):
    """Raised when privacy constraints are violated."""

    pass


class CompressionError(FLError):
    """Raised when gradient compression fails."""

    pass


class ClientError(FLError):
    """Raised when client operations fail."""

    pass


class ServerError(FLError):
    """Raised when server operations fail."""

    pass
