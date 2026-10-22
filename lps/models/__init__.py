"""模型层：改进结构与注意力模块。"""
from .attention import CBAM, CoordAtt, ECA, inject_modules
from .builder import build, param_summary, write_cfg

__all__ = ["CBAM", "CoordAtt", "ECA", "inject_modules", "build", "param_summary", "write_cfg"]
