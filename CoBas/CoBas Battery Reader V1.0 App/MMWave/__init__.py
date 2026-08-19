# ruff: noqa: N999
"""CoBas bridge for the IWR6843AOP range-angle capture pipeline."""

from .capture import (
    MMWaveCaptureEvent,
    MMWaveCaptureService,
)

__all__ = ("MMWaveCaptureEvent", "MMWaveCaptureService")
