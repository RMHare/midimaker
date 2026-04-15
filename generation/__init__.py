"""
generation/__init__.py
======================
Generation subsystem for MidiMaker.

Exposes GenerationRequest, GenerationResult, and BaseGenerator for use
across all generator implementations.
"""

from generation.base import BaseGenerator, GenerationRequest, GenerationResult

__all__ = ["BaseGenerator", "GenerationRequest", "GenerationResult"]
