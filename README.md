# 牲畜姿态识别与行为分析系统

在老师给出的 YOLOv8 羊行为识别框架基础上扩展而来。相比原始框架，主要改动集中在三处：

1. **网络结构**：在 backbone 末端、P3、P4 三处插入 CBAM 注意力，检测头类别数由数据配置驱动
2. **类别体系**：支持多套类别方案，其中 `pose_view8` 利用原始文件名中的拍摄视角信息，把 4 类行为扩展为 8 类（行为 × 视角），无需新增人工标注
3. **软件形态**：从三个命令行脚本扩展为带界面的完整系统（数据集管理 / 训练 / 评估 / 推理 / 报告导出），可打包为免环境安装的 exe

## 快速开始

```bash
# 1. 准备数据（按类别方案派生训练视图）
python scripts/prepare_data.py --all

# 2. 数据自检（原始框架 README 提到但未提供该脚本，此处补实现）
python check_dataset.py --verify-images 20

# 3. 训练
python train.py --scheme behavior4 --epochs 120 --batch 8

# 4. 验证（--split test 用全程未参与训练的测试集）
python val.py --split test --scheme behavior4

# 5. 推理
python predict.py --source 图片或视频路径

# 6. 打开图形界面
python app.py
```

也可以直接用界面完成全部流程，无需敲命令。

## 目录结构

```
livestock_posture/
├─ app.py                  图形界面入口
├─ train.py                训练入口（与原始框架同名，参数兼容）
├─ val.py                  验证入口
├─ predict.py              推理入口
├─ check_dataset.py        数据自检
├─ data.yaml               数据集配置
├─ build.spec              PyInstaller 打包配置
├─ configs/
│  ├─ class_scheme.yaml    类别方案定义（行为 / 视角组合规则）
│  └─ data_*.yaml          各方案派生的数据配置（自动生成）
├─ lps/
│  ├─ config.py            路径、类别方案、训练超参
│  ├─ data/                自检、类别重映射、统计、属性派生、毛色标注
│  ├─ models/              注意力模块、改进结构、模型构建
│  ├─ core/                检测引擎
│  ├─ services/            绘图、报告导出
│  └─ ui/                  PyQt6 界面（四个页面 + 通用组件）
└─ scripts/                可独立执行的任务脚本
```

## 类别方案

| 方案 | 类别数 | 说明 |
|---|---|---|
| `behavior4` | 4 | 采食 / 卧息 / 行走 / 奔跑，与原始数据一致 |
| `activity3` | 3 | 合并行走与奔跑为"活动"，缓解长尾与单帧歧义 |
| `pose_view8` | 8 | 行为 × 视角（正面/侧面），视角取自原始文件名 |

`pose_view8` 的依据：原始图像命名为 `<编号>-<行为>-<视角>-<来源>-<帧号>`，
视角覆盖 7198/7390 个标注框（97.4%），属于数据集作者的原始信息，不需要补标。
无法判断视角的 50 张图会被整张跳过，避免污染标签。

## 关于毛色

毛色**不作为检测类别**。8 类 × 6 色会让每类样本过少，训练必然崩。
毛色由 `lps/data/color_labeler.py` 自动标注（分割取色，实测约 85% 准确），
输出 `outputs/attributes/color_labels.csv`，带置信度，低置信度样本可过滤。

## 对照原始框架

| 原始框架 | 本项目 |
|---|---|
| `train.py` / `val.py` / `predict.py` | 同名保留，参数兼容，内部改为改进模型 |
| `data.yaml` | 保留，并支持多套类别方案派生 |
| `check_dataset.py`（README 提到但缺失） | 已补实现 |
| 固定 4 类、原生 yolov8n | 可切换类别方案 + CBAM 改进结构 |
| 仅命令行 | 命令行 + 图形界面 + 免环境 exe |
