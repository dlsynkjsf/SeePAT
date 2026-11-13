"""Compatibility imports for historical video-processing tools."""

from seepat.preprocessing.processor import VideoProcessor

PilotVideoProcessor = VideoProcessor

__all__ = ["PilotVideoProcessor", "VideoProcessor"]
