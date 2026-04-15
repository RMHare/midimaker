"""MidiMaker GUI controllers — mediators between screens and core modules."""

from app.controllers.project_controller import ProjectController
from app.controllers.generation_controller import GenerationController
from app.controllers.training_controller import TrainingController

__all__ = ["ProjectController", "GenerationController", "TrainingController"]
