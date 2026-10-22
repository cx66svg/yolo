# 目标检测训练与推理平台

基于改进 YOLOv8 的通用目标检测平台。导入任意 YOLO 格式数据集即可完成训练、评估与推理，
**类别数量由数据配置实时决定**，不再局限于固定类别。项目起源于羊行为识别，现已泛化为
通用检测工具，并提供图形界面与免环境 exe。

## 核心特性

- **任意数据集、任意类别数**：标准 YOLO 格式（`images/` + `labels/` + `data.yaml`，
  含 `train`/`val`/`test` 划分）导入即可，类别数由 `data.yaml` 的 `names` 决定，不写死。
- **改进网络结构**：在 backbone 末端与 P3、P4 三处插入 CBAM 注意力，检测头类别数由数据驱动。
- **迁移学习**：自动加载官方 `yolov8n.pt` 预训练权重，加速收敛。
- **完整流程**：数据集导入与自检 → 训练 → 评估 → 推理 → 报告导出，全流程覆盖。
- **多输入推理**：单张图片、图片目录、视频文件、摄像头实时检测。
- **免环境部署**：PyInstaller 打包为 Windows exe，双击即用，无需安装 Python / CUDA。

## 界面功能

| 页面 | 功能 |
|---|---|
| 项目管理 | 项目建档、数据集导入 |
| 数据集管理 | 数据集校验、train/val/test 划分、类别方案 |
| 模型训练 | 改进结构训练，超参可调，实时日志与曲线 |
| 模型评估 | mAP@0.5 / mAP@0.5:0.95、逐类 AP、混淆矩阵 |
| 检测推理 | 图像/视频/摄像头检测，标注图、明细 CSV、统计报告 |

## 快速开始

### 图形界面

```bash
python app.py
```

或直接双击打包好的 exe。按「项目管理 → 导入数据集 → 模型训练 → 模型评估 → 检测推理」
的顺序操作即可完成全部流程。

### 命令行

```bash
# 训练（--data 指定数据集配置，类别数由数据决定）
python train.py --data 你的数据.yaml --epochs 120 --batch 8

# 验证（--split test 用全程未参与训练的测试集）
python val.py --weights outputs/runs/<实验名>/weights/best.pt --data 你的数据.yaml --split test

# 推理
python predict.py --source 图片或视频路径 --weights outputs/runs/<实验名>/weights/best.pt

# 数据自检
python check_dataset.py --verify-images 20
```

## 训练自定义数据集

1. 准备 YOLO 格式数据集，目录形如：

```
你的数据集/
├─ data.yaml          # 声明 train/val/test 路径与 names（类别名）
├─ train/
│  ├─ images/
│  └─ labels/
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

2. 在 `data.yaml` 中声明 `names`（类别名列表），类别数不限。
3. 在软件「项目管理」中导入，或直接 `python train.py --data 你的数据集/data.yaml`。

## 目录结构

```
├─ app.py                  图形界面入口
├─ train.py / val.py / predict.py   训练 / 验证 / 推理入口
├─ check_dataset.py        数据自检
├─ build.spec              PyInstaller 打包配置
├─ configs/                预设类别方案与派生数据配置
├─ lps/
│  ├─ config.py            路径、超参、预设类别方案
│  ├─ data/                数据校验、类别重映射、统计、属性派生
│  ├─ models/              CBAM 注意力、改进结构、模型构建
│  ├─ core/                检测引擎、项目建档
│  ├─ services/            绘图、报告导出
│  └─ ui/                  PyQt6 界面
└─ scripts/                可独立执行的任务脚本
```

## 技术说明

- **CBAM 注意力**：在 backbone 与颈部关键位置插入通道 + 空间注意力，增强特征表达。
- **迁移学习**：改进结构的共享层直接继承官方预训练权重，新增层随机初始化后由训练学习。
- **类别数数据驱动**：模型检测头类别数由 `data.yaml` 的 `names` 长度决定，切换数据集无需改代码。

## 预设类别方案（示例，非限制）

项目早期针对羊行为数据集提供了三套示例方案，仅为演示类别派生能力，**不代表平台的能力上限**：

| 方案 | 类别数 | 说明 |
|---|---|---|
| `behavior4` | 4 | 采食 / 卧息 / 行走 / 奔跑 |
| `activity3` | 3 | 合并行走与奔跑为"活动" |
| `pose_view8` | 8 | 行为 × 视角，视角取自原始文件名 |

任意数据集均可直接导入训练，无需预先定义方案。

## 可选：目标属性派生（动物场景）

检测框几何信息可派生姿态、遮挡等属性；`lps/data/color_labeler.py` 提供毛色自动标注
（GrabCut 取色）。此为动物场景的可选增强功能，不影响通用检测流程。
