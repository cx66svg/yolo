"""核心层：检测推理与项目建档。"""
from .detector import Detection, Detector, FrameResult, read_model_names
from .project import Project, ProjectStore

__all__ = ["Detection", "Detector",
    "read_model_names", "FrameResult", "Project", "ProjectStore"]
