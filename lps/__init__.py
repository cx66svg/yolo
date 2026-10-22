"""牲畜姿态识别与行为分析系统（Livestock Posture Analysis System）。

分层结构::

    lps.config        全局配置与类别方案
    lps.data          数据集自检、类别重映射、统计分析
    lps.models        改进的网络结构与注意力模块
    lps.core          检测推理引擎
    lps.services      行为分析与报告导出
    lps.ui            PyQt6 桌面端界面
"""

__version__ = "1.0.0"
__app_name__ = "目标检测训练与推理平台"
