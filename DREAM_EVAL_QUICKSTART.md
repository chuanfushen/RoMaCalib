# DREAM 评测上手指南

本文档覆盖 RoMaV2 DREAM 评测所需的安装、数据和模型下载、MuJoCo 预渲染以及批量评测。所有可变路径和主要评测参数集中在 [`config/dream_eval.toml`](config/dream_eval.toml)。

DREAM 的可复用实现位于 `src/romav2/benchmarks/utils/`；`scripts/` 只保留下载和统一命令行入口。

## 1. 安装环境

在仓库根目录运行：

```bash
uv sync --extra eval --extra setup
```

`eval` 包含 OpenCV 等评测依赖，`setup` 包含 ModelScope 和 `gdown` 下载工具。

Franka Panda 的 MJCF、mesh 和许可证已经包含在：

```text
assets/third_party/mujoco_menagerie/franka_emika_panda/
```

## 2. 修改统一配置

默认配置文件是：

```text
config/dream_eval.toml
```

其中：

- `[paths]`：数据、SAM3、MJCF、预渲染及评测输出路径。
- `[datasets.*]`：DREAM 数据集名称、目录和 Google Drive ID。
- `[evaluation]`：评测数据集、采样数、视角数、batch size、设备和 mask 参数。
- `[prerender]`：MuJoCo 渲染分辨率与相机参数。

相对路径均相对于仓库根目录解析。若不想修改仓库配置，可以复制一份：

```bash
cp config/dream_eval.toml config/dream_eval.local.toml
export DREAM_EVAL_CONFIG=config/dream_eval.local.toml
```

以下路径也可用环境变量临时覆盖：

```text
DREAM_DATA_ROOT
DREAM_SAM3_DIR
DREAM_SAM3_CHECKPOINT
DREAM_MUJOCO_XML
DREAM_PRERENDER_ROOT
DREAM_OUTPUT_ROOT
DREAM_REAL_DATASET
DREAM_DR_DATASET
DREAM_PHOTO_DATASET
```

例如：

```bash
export DREAM_DR_DATASET=/datasets/dream/panda_synth_test_dr
export DREAM_OUTPUT_ROOT=/results/romav2_dream
```

## 3. 下载 SAM3 与 DREAM 数据

默认下载配置中启用的 `dr` 和 `photo` 数据集，同时从 ModelScope 下载 `facebook/sam3`：

```bash
uv run --extra setup python scripts/download_dream_assets.py
```

仅下载指定数据集：

```bash
uv run --extra setup python scripts/download_dream_assets.py --datasets dr
```

仅下载数据集、跳过 SAM3：

```bash
uv run --extra setup python scripts/download_dream_assets.py --skip-sam3 --datasets real dr photo
```

重新下载已有内容可增加 `--force`。DREAM 数据 ID 来自 [NVIDIA DREAM 官方下载脚本](https://github.com/NVlabs/DREAM/blob/master/data/DOWNLOAD.sh)，SAM3 来自 [ModelScope 的 facebook/sam3](https://www.modelscope.cn/models/facebook/sam3)。

下载后默认目录结构为：

```text
data/dream/
├── real/panda-3cam_azure/
└── synthetic/
    ├── panda_synth_test_dr/
    └── panda_synth_test_photo/

assets/checkpoints/sam3/
└── sam3.pt
```

SAM3 的 BPE tokenizer 资源由已安装的 `sam3` package 自动解析，不需要单独下载或配置。

## 4. 预渲染 MuJoCo 视角

评测前必须为每个 DREAM 帧生成渲染视角和相机参数：

```bash
uv run --extra eval python scripts/run_dream_eval.py --prerender
```

只预渲染一个数据集：

```bash
uv run --extra eval python scripts/run_dream_eval.py --prerender --datasets dr
```

输出位于配置中的 `paths.prerender_root`。每帧应包含成对文件，例如：

```text
000005/view_00.png
000005/view_00_camera.npz
```

`evaluation.views` 改变后需要用相同视角数重新预渲染。

预渲染通过 `[prerender].workers` 做多进程帧分片，默认值为 `4`。每个进程使用独立的 MuJoCo model、data、renderer 和 EGL context；不要让多个 Python 线程共享同一个 renderer。单 GPU 建议从 `2` 或 `4` 开始，根据 GPU 利用率、显存和磁盘写入速度调整：

```toml
[prerender]
workers = 4
```

如果出现 EGL context、显存不足或驱动错误，将它降为 `1`。

## 5. 运行 batch 评测

```bash
bash scripts/eval_dream.sh
```

也可以直接调用配置入口：

```bash
uv run --extra eval python scripts/run_dream_eval.py --datasets dr photo
```

只检查配置展开后的命令而不执行：

```bash
uv run --extra eval python scripts/run_dream_eval.py --dry-run
```

批量推理大小由以下配置控制：

```toml
[evaluation]
views = 6
match_batch_size = 6
```

启用输入 mask：

```toml
[evaluation]
mask_input = true
mask_prompt = "robotic arm"
```

SAM3 生成的 mask 保存在每帧输出目录的 `input_mask.png`。如不使用 mask，将 `mask_input` 改为 `false`。

### 按仓库 benchmark 测试风格运行

DREAM 和 Mega1500/ScanNet1500 一样提供 `Benchmark.benchmark(model)` 接口，三个数据集对应三个 pytest 测试：

```bash
uv run --extra dev --extra eval pytest tests/test_dream.py -v
```

也可以单独运行：

```bash
uv run --extra dev --extra eval pytest tests/test_dream.py::test_dream_real -v
uv run --extra dev --extra eval pytest tests/test_dream.py::test_dream_dr -v
uv run --extra dev --extra eval pytest tests/test_dream.py::test_dream_photo -v
```

直接在 Python 中使用 benchmark：

```python
from romav2 import RoMaV2
from romav2.benchmarks import Dream

model = RoMaV2()
model.apply_setting("precise")
benchmark = Dream(
    data_root="data/dream/synthetic/panda_synth_test_dr",
    prerender_root="outputs/dream_mujoco_prerendered_views/panda_synth_test_dr",
    mujoco_xml="assets/third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
    output_dir="outputs/dream_mujoco_match_eval_batch/panda_synth_test_dr_sample300",
)
metrics = benchmark.benchmark(model)
```

## 6. 推荐执行顺序

```bash
# 1. 安装
uv sync --extra eval --extra setup

# 2. 检查并修改 config/dream_eval.toml

# 3. 下载配置中启用的数据集和 SAM3
uv run --extra setup python scripts/download_dream_assets.py

# 4. 预渲染配置中 evaluation.datasets 指定的数据集
uv run --extra eval python scripts/run_dream_eval.py --prerender

# 5. 批量评测
bash scripts/eval_dream.sh
```

如果出现 `Missing pre-rendered view artifacts`，说明第 4 步未执行、目录配置不一致，或预渲染时的 `views` 少于评测配置。
