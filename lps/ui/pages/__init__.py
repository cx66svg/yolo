"""按功能分组的界面页面。"""
from .dataset_page import DatasetPage
from .eval_page import EvalPage
from .predict_page import PredictPage
from .project_page import ProjectPage
from .train_page import TrainPage

__all__ = ["DatasetPage", "TrainPage", "EvalPage", "PredictPage", "ProjectPage"]
