"""Errors for SpinPath."""

from __future__ import annotations


class SpinPathException(Exception):
    """Base exception for SpinPath."""


class InvalidModelConfiguration(SpinPathException):
    """Invalid model configuration."""


class InvalidRegistryConfiguration(SpinPathException):
    """Invalid model zoo registry configuration."""
