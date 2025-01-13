"""Errors for WSInfer MIL."""

from __future__ import annotations


class WSInferMILException(Exception):
    """Base exception for WSInfer MIL."""


class InvalidModelConfiguration(WSInferMILException):
    """Invalid model configuration."""


class InvalidRegistryConfiguration(WSInferMILException):
    """Invalid model zoo registry configuration."""
