# DROID-R10-v1 Experiment Tracker

状态更新时间：2026-07-22
当前阶段：三轮迭代 + CalibAll 协议已形成，尚未实现/运行 DROID 正式实验。

## 固定输入

| Item | Value | Status |
|---|---|---|
| Dataset | `/data1/scf/Generative/data1/datasets/droid_raw_random10_seed20260722` | verified |
| Sampling seed | `20260722` | frozen |
| Episodes | 10 success episodes | frozen |
| External sessions | 20 (`10 × 2`) | pending semantic camera audit |
| Candidate frames | 2613/camera stream; 5226 over two external views | verified from MP4 |
| Primary split | `frame_index % 5 == 0` held-out | proposed/freeze before run |
| Default calibration frames | 16/session | proposed/freeze before run |
| Refinement iterations | run exactly 3 and retain `T0..T3`; select validation-best per session | frozen |
| CalibAll candidate budgets | 200 / 500 / 1000 steps | frozen for small-sample tuning only |
| Config selection split | 2 episodes × 2 cameras; calibration-only fit/validation | freeze manifest before masks/metrics |
| Final config selection | validation macro IoU, Q10, fallback/failure, runtime (lexicographic) | frozen |
| SAM prompt | `robotic arm` | frozen from current project |
| Bootstrap seed | `20260717` | frozen |
| Local branch | `codex/panda-real-gate-v1-sz` | observed |
| Local HEAD at planning time | `0beec3ee3ca4cab99444fee6666bf42b0ca0a794` | observed, not experiment commit |

## Milestones

| ID | Milestone | Server/GPU | Commit | Output | Status | Evidence / blocker |
|---|---|---|---|---|---|---|
| M0.1 | HDF5 schema + UUID/serial audit | cf / CPU | planning HEAD only | `0012_droid_raw_r10_overlay_smoke_TRI_20260722` | partial | TRI episode schema and serial mapping verified; remaining 9 pending |
| M0.2 | Intrinsics/extrinsics convention audit | cf / CPU | planning HEAD only | same | partial | official 1280x720 intrinsics loaded; raw camera-to-base smoke rendered; all sessions pending |
| M0.3 | MP4-state synchronization audit | cf / CPU | TBD | TBD | pending | must test index vs timestamp |
| M0.4 | Panda+Robotiq model audit | local + cf GPU 0 EGL | planning HEAD only | `.../frame0030_ext1_rawcalib_realgripper` | partial | +90 deg mount, normalized raw gripper mapping, and real-like dark material visually plausible; Git import and multi-pose validation pending |
| M1.1 | DROID adapter/config implementation | local | TBD | n/a | pending | no DROID runner exists locally |
| M1.2 | Metrics + hierarchical bootstrap tests | local | TBD | n/a | pending | IoU failure denominator must be tested |
| M1.3 | Visualization implementation | local | TBD | n/a | pending | native + 320×180 outputs |
| M1.4 | ruff/pytest/dry-run | local | TBD | n/a | pending | |
| M2 | 1 session × 5-frame, 3-iteration CalibAll smoke | cf / one idle GPU | TBD | TBD | pending | verify T0..T3 artifacts and fallback |
| B1/M3 | Small-sample config ablation | cf / one idle GPU | TBD | TBD | pending | S0, R3-G, S0+C500, R3-G+C3 at 200/500/1000 |
| B2 | Final main comparison | cf / one idle GPU | same frozen commit | TBD | pending | raw calib, S0, R3-G, selected R3-G+C3 |
| B3 | Iteration/CalibAll contribution | cf / one idle GPU | same as B1 | reuse B1 | pending | report all T0/T1/T2/T3 candidates and selected iteration |
| B4 | Model/pseudo-GT audit | cf + manual review | same as B1 | TBD | pending | 100-frame blind audit |
| M5.1 | Independent metric recomputation | cf / CPU | same as B1 | TBD | pending | |
| M5.2 | Figures, tables, videos | cf / CPU/GPU as needed | same as B1 | TBD | pending | |

## Required result checks

| Check | Expected | Actual | Status |
|---|---:|---:|---|
| Parsed episodes | 10 | | pending |
| External camera sessions | 20 | | pending |
| Wrist sessions in primary metric | 0 | | pending |
| Held-out manifest overlap with calibration | 0 | | pending |
| Held-out masks read during config selection | 0 | | pending |
| Refinement iterations saved per solved session | 4 poses (`T0..T3`) | | pending |
| Final pose selection uses held-out masks | no; calibration-validation only | | pending |
| Selected-iteration distribution reported | T0/T1/T2/T3 session counts | | pending |
| Selected config hash frozen before final run | yes | | pending |
| SAM mask manifest/hash coverage | 100% files accounted for | | pending |
| Requested sessions in summary | 20/system | | pending |
| Failed solve retained in denominator | yes | | pending |
| Native evaluation resolution | 1280×720 | | pending |
| Paper-compatible resolution | 320×180 | | pending |
| Main metric independent recomputation match | exact/tolerance documented | | pending |

## Current blockers and decisions

1. cf 当前 RoMaCalib `.venv` 没有 `h5py`；应通过本地 dependency/lockfile 变更解决。
2. cf 没有 ZED SDK/`pyzed`；主实验使用 DROID 官方 `intrinsics.json`，SVO 读取只做可选审计。
3. 2F-85 上游资产和 DROID 装配关系已定位，但尚未进入本地 RoMaCalib Git；正式 full-body
   IoU 前需连同 LICENSE 导入，并验证 `[0,0,0.107] m + yaw +90 deg` 和 gripper mimic mapping。
4. cf 仓库存在大量既有修改/未跟踪文件；不得在该工作树修改或同步代码。正式部署前需用户处理，
   或指定一个干净且不覆盖历史结果的 Git checkout/worktree。
5. CtRNet-X 的 10 个 episode 已不可得，且公开协议不完整；文献 0.8356 只做 contextual reference。
6. 现有 `run_caliball_mvp.py` 是单帧 silhouette MVP；DROID 正式方法需要在本地实现共享
   6DoF pose 的多帧 loss 聚合与跨帧 validation gate，不能直接套用单帧脚本。

## Next action

用户确认该协议后，进入 M0：先下载很小的官方 `intrinsics.json`/calibration JSON 到 cf 数据目录，
在本地实现 HDF5 adapter 与正确的 Panda+Robotiq 模型支持，再通过 Git 部署 smoke test。
