# DROID CtRNet-X-like Experiment Plan (DROID-R10-v1)

状态：三轮迭代 + CalibAll 方案冻结候选，尚未实现、尚未运行 DROID GPU 实验
日期：2026-07-22
目标服务器：cf / server A
数据根目录：`/data1/scf/Generative/data1/datasets/droid_raw_random10_seed20260722`
抽样种子：`20260722`

## 1. 先复述 CtRNet-X 实际做了什么

CtRNet-X 在 DROID 上不是用 6DoF 外参 GT 做位姿误差，而是用渲染轮廓与 SAM
机器人掩码的 IoU 做间接评测：

- 随机选择 10 个 DROID video episodes，共报告 3232 帧；
- 从原始 SVO 读取每个相机的内参，从 `trajectory.h5` 按帧索引读取 7 个关节角；
- 对一段视频中所有可靠关键点做汇总，只求一个共享的固定相机外参；
- 用该外参逐帧渲染 Panda，并与 SAM 掩码计算 IoU；
- 论文报告 CtRNet-X 平均 IoU 0.8356，DROID 原始外参渲染 IoU 0.0186；
- 官方代码把图像缩放到 320×180，按帧累计 IoU；DROID 入口只通过相机文件名规则选相机，
  没有按语义显式区分 `exterior_1`、`exterior_2` 和 wrist；
- 官方代码的渲染 mesh 列表只包含 Panda `link0..link7`，没有 Robotiq 2F-85；
- 论文和仓库未冻结公开的 10 个 episode ID、SAM 版本/提示词、人工 mask 质检、
  episode 宏平均或校准/评测帧隔离方式。

因此，0.8356 只能作为文献参考值，不能与我们的随机 10 条 episode 直接做同分布比较。

参考：

- Paper: https://arxiv.org/pdf/2409.10441
- Code: https://github.com/darthandvader/CtRNet-X
- DROID annotations: https://huggingface.co/KarlP/droid

## 2. 冻结的研究问题与 claims

最多保留两个 claims。

### Claim C1：跨帧共享外参的有效性

在每个 DROID episode 的固定外部相机上，RoMaCalib 使用少量校准帧估计一个共享的
`T_camera<-robot_base`，能在未参与估计的 held-out 帧上获得比 DROID 原始标定和
RoMaCalib 单帧/一阶段 S0 更高的机器人轮廓 IoU。

### Claim C2：三轮闭环与 CalibAll 精修的有效性和安全性

相对 S0 和不使用 CalibAll 的三轮闭环，三轮 `RoMaCalib update -> CalibAll` 在跨帧验证
gate 约束下进一步提高 held-out IoU，并降低 episode-camera 级 `Delta IoU < -0.05` 的比例。

不做以下 claims：

- 不把 SAM3 mask 称作人工 GT；统一称 `SAM3 pseudo-GT`；
- 不把原始或 post-hoc DROID 外参称作无噪声 6DoF GT；它们只是官方标定 baseline；
- 不把 DROID mask IoU 与 DREAM/Baxter 的 2D、3D、ADD、PCK 指标横向比较；
- 不宣称直接复现 CtRNet-X 的 0.8356，因为 episode、SAM、机器人模型和聚合口径不同。

## 3. 数据冻结

### 3.1 样本

使用已经从官方 `gs://gresearch/robotics/droid_raw/1.0.1` 随机抽取并校验的 10 条
success episodes。每条均有 metadata JSON、`trajectory.h5`、3 个非 stereo MP4、
3 个 SVO，官方对象大小和 GCS MD5 均通过。

候选帧数：

- 每个相机：2613 帧；
- 两个固定外部相机：5226 个 episode-camera-frame；
- wrist 相机：不进入主实验，因为其 `T_camera<-base` 随机器人关节运动，不满足固定外参假设。

数据校验记录：

- `selection_manifest.json` SHA256:
  `b54761d043d1c63e580e356b0e2fea56555699998535f131183069f4cc1e65dd`
- `verification_summary.json` SHA256:
  `3c418a33ae864a1afbe40a5e3c0e8219559e73b2458ec548f45d73b90601e57`
- `source_manifest.json` SHA256:
  `f540f0a86e791d11588670f82e8045b54ae79959a8104aa6781087486f65a5d2`

### 3.2 相机与同步

每条 episode 只使用 metadata 显式标记的两个 external camera serial，不能按文件名排序猜测。

- 原始 baseline 外参：`trajectory.h5/observation/camera_extrinsics/<serial>_left[0]`；
- 首选内参：DROID 官方 `intrinsics.json`，按 metadata episode UUID 和 camera serial 查询；
- 一致性审计：若后续 cf 可用 ZED SDK，再从 SVO 读取内参并比较；SDK 不作为主实验阻塞项；
- 可选 reference：官方 post-hoc `cam2base_extrinsics.json` / superset；仅在命中该 UUID 时报告，
  同时报告覆盖率，不能用缺失 episode 的子集代替主分母。

同步在 M0 中冻结。初始假设为 CtRNet-X 官方代码的 index alignment：MP4 frame `i`
对应 HDF5 state `i`。必须先满足：

1. 每个 episode-camera 的 MP4 帧数与 joint state 数差不超过 1；
2. 首、中、末各至少一帧的官方外参投影/渲染没有明显的时间错位趋势；
3. 若存在 camera/state timestamps，则独立核对 index alignment 的最大时间差。

任一条件失败时，停止正式实验，改为 timestamp 最近邻同步并生成新的 manifest。

### 3.3 机器人模型

DROID 实机是 Franka Panda + Robotiq 2F-85。当前 RoMaCalib Git 资产只有 Panda 原厂
hand，但 cf 的 Generative 第三方目录已经保存了可复用的模型来源：

- MuJoCo Menagerie `robotiq_2f85`，上游 commit
  `accb6df40a9a1d1e49eff88157f6818b63a49335`，BSD-2-Clause；
- DROID/Franka external-gripper xacro；
- 已组合的 `droid_menagerie_panda_arg85.xml` 与 `droid_menagerie_panda_2f85.xml`。

这里的 `arg85` 是 Robotiq ARG2F-85 的模型命名，不是另一种 DROID gripper。

冻结装配链：

1. `panda_link7 -> panda_link8`：固定平移 `[0, 0, 0.107] m`；
2. `panda_link8 -> robotiq_arg2f_base_link`：固定平移 `[0, 0, 0]`，绕 z 轴 `+90 deg`；
3. 该变换来自现有 `robotiq_mounted_on_panda.urdf.xacro`，并与 DROID 专用组合 MJCF 一致；
4. Menagerie 原生 `2f85.xml` 额外包含 `base_mount` 和内部 mesh-frame 旋转，不能只比较
   单个 quaternion 就判断与 ARG85 模型不一致，必须比较最终 visual mesh pose。

关节映射冻结为：

- HDF5 前 7 维按名称写入 Panda `joint1..joint7`；
- DROID gripper opening 先裁剪到 `[0, 0.04] m`；
- ARG85 主驱动角 `q = (0.04 - opening) / 0.04 * 0.725 rad`，其余关节按 mimic 符号联动；
- 若采用 Menagerie linkage profile，则用同一 normalized closure 在已验证的 open/closed
  qpos 之间插值，不能把毫米 opening 直接写入旋转关节。

外观材质冻结为：Panda 使用接近实机的白色/深灰色，Robotiq 2F-85 使用接近实机的
深灰/黑色。彩色 finger/knuckle 材质只允许用于内部装配诊断，不进入 RoMaV2 matching、
定量 mask 或论文可视化。

冻结两个 rendering profiles：

- `droid_full`（主结果）：Panda 7 关节 + Robotiq 2F-85，使用 HDF5 gripper state；
- `panda_arm_only`（paper-compatible / model ablation）：只渲染 `link0..link7`，接近
  CtRNet-X 公开代码。

正确的 Robotiq MJCF/mesh、LICENSE、上游 commit 和装配文件必须作为小型源码资产在本地
加入 Git，再通过 Git 同步；不得直接引用 Generative 的主机绝对路径完成正式结果。

## 4. 帧划分与防泄漏

每个 `episode × external camera` 是一个独立 session，共 20 个 sessions。

### 主协议：held-out

- 评测帧：`frame_index % 5 == 0`，约 20%，贯穿整段动作；
- 校准候选帧：其余约 80%；
- 默认校准预算：从候选帧中按时间均匀取 16 帧；
- SAM3 mask、RoMaV2 correspondence、gate、PnP/refinement 只能读取校准帧；
- held-out mask 仅在最终 pose 冻结后用于评测；
- 若 session 求解失败，则该 session 所有 SAM-valid held-out 帧 IoU 记 0，不能删除。

### 小样本配置选择：calibration-only tuning split

- 固定选取 2 个 episode、两个 external cameras，共 4 sessions；在看到任何 held-out 指标前
  将 episode ID 和帧号写入 `tuning_manifest.json`；
- 外层 held-out 仍为 `frame_index % 5 == 0`，绝不参与配置选择；
- 其余 calibration pool 内再做嵌套划分：每个 session 时间均匀取 16 个 fit frames 和
  8 个 disjoint validation frames；SAM3 mask 可用于 fit/validation，但不能读取 held-out mask；
- 配置排序采用预先冻结的词典序：validation session-macro IoU 最大，其次 Q10 最大，
  再其次失败/回退率最低，最后 runtime 最低；不是从最终 10-episode 结果中挑最好配置；
- 选中配置后记录完整 TOML、Git commit 和 SHA256，随后冻结。最终 10 episodes 只运行一次，
  无论结果好坏均按完整分母报告；若因实现 bug 重跑，必须升级实验 ID 并披露原因。

### Paper-compatible 描述性协议

- 使用 `exterior_1_left` 的全部 2613 帧；
- 在 320×180 上计算 frame-weighted IoU；
- 允许像 CtRNet-X 一样用全视频估计并在同视频评测；
- 单独标为 `descriptive / non-held-out`，不用于支持 C1/C2。

## 5. 系统与实验块（compact plan）

### B0. Data/model/convention audit（必须先通过）

- Claim：不直接支持论文 claim，是所有结果的有效性前提。
- 数据：10 episodes，先审计 episode 1，再批量审计 10 条。
- 检查：UUID/serial 映射、HDF5 字段、MP4-state 同步、内参顺序 `[fx,cx,fy,cy]`、
  外参方向 `camera->base`、Euler `xyz`、分辨率缩放、Panda joint 名称映射、gripper state。
- 输出：`droid_manifest.json`、`camera_audit.json`、`model_audit.json`、6 张 sanity overlays。
- 通过：20/20 external sessions 可解析；无未知单位/方向；完整模型单帧 EGL 渲染通过。
- 失败：任一 session 相机身份或同步不确定，不得进入 B1。

### B1. Small-sample method/config ablation（先执行，必须）

- 数据：上述 2 episodes × 2 cameras 的 calibration-only tuning split；不读取 held-out mask。
- 所有候选固定执行 3 轮并保存 `T0/T1/T2/T3`。每个 session 的最终 pose 从四个候选中
  按 calibration-validation score 选择最佳轮次；held-out mask 严禁参与选轮。gate 拒绝更新时
  该轮保持上一有效 pose，并继续记录状态。
- 方法消融：
  1. `S0`；
  2. `R3-G`：三轮 RoMaCalib 闭环，无 CalibAll；
  3. `S0+C500`：只在 S0 后做一次 CalibAll，隔离可微轮廓精修贡献；
  4. `R3-G+C3`：每轮 RoMaCalib 更新后均做 CalibAll，是候选主方法。
- CalibAll budget 消融：只对 `R3-G+C3` 比较 `200/500/1000` steps；沿用现有
  `integration_v0`（Adam, lr=5e-4, weight decay=1e-6, gradient clip=1.0,
  cosine warm restarts, best-loss checkpoint），不把作者原始 10k-step 配方扩展到正式网格。
- 每次 CalibAll 的输入是当轮 RoMaCalib pose；fit mask 只来自 16 个 fit frames。接受条件：
  数值有效、renderer replay IoU >= 0.95、位移 <= 0.20 m、旋转 <= 20 deg，并且 8 个
  disjoint calibration-validation frames 的 macro IoU 相对进入该轮前不下降。否则回退。
- 配置级选择输出：`selected_config.toml/json`、候选配置排名、每轮 validation IoU、pose delta、
  accepted/rejected reason、runtime。胜出配置冻结后进入 B2；正式实验仍允许每个 session
  按同一条预注册 validation 规则在 `T0..T3` 中选择不同的最佳轮次。

### B2. Final main comparison（必须）

- Claims：C1、C2。
- 数据：20 sessions，held-out protocol，默认 16 calibration frames/session。
- 系统：
  1. `DROID-raw-calib`：原始 HDF5 外参；
  2. `S0`：现有 RoMaCalib 一阶段；
  3. `R3-G`：固定三轮 RoMaCalib 闭环，无 CalibAll；
  4. `R3-G+C3(best)`：固定三轮、每轮 CalibAll，使用 B1 选中并冻结的 budget/config。
- 所有 RoMaCalib 系统每个 session 只能输出一个共享外参。
- 主指标：native 1280×720 held-out SAM3 pseudo-GT mask IoU，先帧平均到 session，
  再 camera 平均到 episode，最后对 10 episodes 做 macro mean。
- 次指标：median/Q10 IoU、mask precision/recall、boundary F-score、solve failure rate、
  SAM-valid coverage、runtime、每 session 接受/拒绝状态计数。
- 统计：paired hierarchical bootstrap，先 bootstrap episode，再保留其两个 cameras；
  10,000 次，seed `20260717`，报告 Delta IoU 及 95% CI。
- C1 支持条件：`R3-G+C3(best)` 相对原始标定与 S0 的 macro Delta IoU 均为正，且
  95% CI 下界 > 0。
- C2 支持条件：`R3-G+C3(best)` 相对 R3-G 的 macro IoU 为正，且相对 S0 的
  `Delta IoU < -0.05` session 比例不超过 10%；同时报告三轮中所有 gate 回退。
- 否证：CI 跨 0、收益只来自少数长 episode、或 R3-G catastrophic regression 不降。
- 论文表：主表按系统报告 macro IoU、Q10、boundary F、failure、coverage、runtime。

### B3. Iteration / CalibAll contribution（必须，小样本消融复用）

- 在 B1 的 disjoint validation frames 上统一报告 T0、T1、T2、T3，并报告各轮成为最终
  selected pose 的 session 数量；
- 消融表只保留 `S0 / R3-G / S0+C500 / R3-G+C3(200/500/1000)`，避免无解释的大网格；
- 主方法输出 validation-best iteration，而不是固定 T3。选轮规则和 tie-break 必须在正式
  held-out 运行前冻结：validation IoU 最大；并列时依次选 pose delta 更小、轮次更早者。

### B4. Model fidelity and pseudo-GT audit（必须）

- Claim：排除“高 IoU 只是模型/掩码口径”的替代解释。
- 系统：`droid_full` vs `panda_arm_only`，固定 B1 的 R3-G pose；不重新优化 pose。
- 数据：全部 held-out 帧；另按 session 均匀抽 5 帧，共 100 帧做 blind visual audit。
- 指标：两种 render profile IoU、SAM-valid/明显漏分割/包含非机器人像素比例。
- 支持：完整模型不劣于 arm-only，且 SAM3 pseudo-GT 主要覆盖真实机器人。
- 否证：SAM3 系统性漏掉 gripper/base，或模型差异主导结论；此时正文只报告 arm-only
  compatible 结果，并把 full-body 作为失败分析。
- 论文图：TP 绿、FP 红、FN 蓝的 error maps；100-frame audit montage。

### Nice-to-have（不阻塞主结果）

- 官方 post-hoc calibration baseline（仅覆盖 UUID 命中部分，明确分母）；
- wrist camera 作为“固定外参假设不成立”的负对照；
- SVO 内参与官方 `intrinsics.json` 的数值一致性；
- 额外随机 seed 的第二批 10 episodes，用于检验样本波动。

## 6. SAM3 pseudo-GT 冻结

- 模型/checkpoint/hash沿用当前 RoMaCalib 已验证的 SAM3；
- prompt 固定为 `robotic arm`，不针对这 10 条 episode 调 prompt；
- mask 生成一次并缓存，记录 checkpoint hash、prompt、输入 frame hash、输出 mask hash；
- 空 mask、多个候选、fallback、mask area ratio 全部写入逐帧 JSON；
- SAM-invalid 帧不进入 IoU 分母，但必须报告覆盖率；算法在 SAM-valid 帧上的求解失败记 IoU=0；
- 100 帧人工检查只做 mask 质量分类，不据此修改 mask 或挑选结果。

## 7. 可视化交付

### Figure 1：10-episode qualitative grid

每个 episode 一行，固定选择一个 held-out 帧，列为：

1. RGB；
2. SAM3 pseudo-GT（绿色半透明）；
3. DROID 原始标定（橙色 contour）；
4. S0（青色）；
5. R3-G（紫色）；
6. R3-G+C3(best)（蓝色）。

每格标 episode 短 ID、camera、frame、IoU，不用“GT overlay”措辞。

### Figure 2：paired quantitative plot

- 每个点是一个 episode 的两相机平均 IoU；
- 连接同一 episode 的不同系统；
- 同时给 macro mean 和 95% bootstrap CI；
- 不用长 episode 的帧数给它更大权重。

### Figure 3：temporal stability

选择 best/median/worst 三个 sessions，画全序列 held-out IoU 曲线；标注 calibration frame、
gate reject、fallback 和 robot visible area。固定外参应该在不同关节姿态下保持一致。

### Figure 4：failure/error map

至少覆盖 occlusion、base 不可见、end-effector 不可见、SAM 漏分割、模型 gripper 不匹配：

- TP：绿色；FP：红色；FN：蓝色；
- 并排展示 S0、R3-G、R3-G+C3(best)；
- 附 correspondence/inlier diagnostic inset。

### Video

每个 episode 至少输出 `exterior_1` 的 side-by-side MP4：RGB+SAM contour、原始标定、R3-G；
可选输出 `exterior_2`。统一 15 FPS 逻辑播放，不能沿用容器中可能误导的 60 FPS metadata。

### 3D camera plot（补充材料）

画 robot base、原始 camera frustum、R3-G frustum。只表示两个估计的相对差异，
不把任一 frustum 标为真值。

## 8. 实现范围

预期本地新增/修改（正式实现阶段）：

- `scripts/run_droid_eval.py`
- `scripts/plot_droid_eval.py`
- `src/romav2/droid_config.py`
- `src/romav2/benchmarks/droid.py`
- DROID multi-frame CalibAll adapter：共享一个 6DoF pose，对 16 个不同关节状态/轮廓的 loss
  做 session-level 聚合；现有 `run_caliball_mvp.py` 是单帧 MVP，不能原样当作正式实现
- `src/romav2/benchmarks/utils/` 中最小必要的 dataset adapter / episode aggregation
- `config/droid_eval.toml`（共享模板，不含主机绝对路径）
- `config/droid_eval.local.toml`（cf 本地忽略文件）
- `tests/test_droid_dataset.py`
- `tests/test_droid_metrics.py`
- 正确授权的 Panda+Robotiq MJCF/mesh 源码资产与 LICENSE

新增 `h5py` 应通过本地 `pyproject.toml` + `uv.lock`，不能在服务器临时 pip 安装形成不可复现环境。

输出 schema 至少包括：

- `manifest.json`
- `camera_audit.json`
- `mask_manifest.json`
- `episode_summary.jsonl`
- `frame_summary.jsonl`
- `summary.json`
- `figures/*.png|pdf`
- `videos/*.mp4`
- config 副本、完整命令、commit、环境、GPU、开始/结束时间与退出码。

## 9. 执行阶梯

### M0：只读审计

核对 HDF5 schema、相机映射、同步、内外参和机器人模型。不开 GPU 长任务。

### M1：本地实现与测试

在本地实现 adapter、metrics、plotting；通过 ruff、针对性 pytest、config dry-run。
用户确认后 commit/push 到 `xieting`。

### M2：cf 单 session smoke

1 episode × 1 external camera × 5 calibration/eval frames：解析、SAM3、预渲染、S0、R3-G、
三轮 CalibAll、
summary、overlay 全链路。核对实际 PNG/NPZ/JSON 数量。

### M3：cf 2 episodes / 4 sessions 小样本消融与配置选择

先检查运行稳定性、OOM、EGL、失败原因和输出 schema，再按 B1 的预注册网格选择配置。
若发现实现 bug，修改代码、生成新 commit，并从头重跑全部候选；不能保留修 bug 前结果
混入排名。产出并冻结 `selected_config` 后才能进入 M4。

### M4：10 episodes 正式实验

只用 B1 冻结的胜出配置执行 B2，共享已冻结的 images、SAM masks、prerenders 和 held-out manifest。
每个 block 使用独立 output root。

### M5：独立重算与可视化

从 `frame_summary.jsonl` 独立重算所有指标和分母，生成主表、消融表、4 组图和视频。

## 10. 资源与停止规则

- 初始 GPU：cf 当时空闲的一张 RTX 6000 Ada；当前快照 0..4 均空闲，但正式运行前重查；
- `workers=1`、保守 batch 开始；M2 后才增大；
- 不与已有进程共享 output/prerender/log；不杀进程、不接管 tmux；
- 发现模型错、相机方向错、同步错、SAM coverage < 80%、或 smoke 中多数 IoU 近零，立即停止；
- 不通过调 gate 阈值追逐这 10 条 episode 的最终 IoU；需要改阈值则版本升级为 DROID-R10-v2，
  全部系统与全部 episodes 重跑并披露变化。

## 11. 最终论文表的最小模板

| System | Render profile | Calib frames | Macro IoU ↑ | Q10 IoU ↑ | Boundary F ↑ | Solve fail ↓ | SAM coverage ↑ | Runtime/session ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DROID raw calibration | droid_full | 0 | | | | | | |
| S0 | droid_full | 16 | | | | | | |
| R3-G (3 iterations) | droid_full | 16 | | | | | | |
| R3-G + CalibAll (3 iterations, selected budget) | droid_full | 16 | | | | | | |

所有指标在结果产生前保持空白；不能把文献中的 0.8356 填入本表。

表注必须报告最终选择的 iteration 分布（T0/T1/T2/T3 各有多少 sessions），明确“3 iterations”
表示最多生成三轮候选，而不是无条件采用第三轮。
