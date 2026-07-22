# DROID Franka Panda + Robotiq 2F-85 assets

This directory contains the versioned MuJoCo robot model used by the DROID
camera-calibration experiments.

- `panda_assets/` and the base `panda.xml` structure were imported from
  `google-deepmind/mujoco_menagerie` commit
  `accb6df40a9a1d1e49eff88157f6818b63a49335`. The upstream Panda license is
  preserved as `PANDA_LICENSE` (Apache-2.0).
- `robotiq_arg85/meshes/` was imported from
  `a-price/robotiq_arg85_description` commit
  `a65190bdbb0666609fe7e8c3bb17341e09e81625`. Its ROS package metadata declares
  the asset package as BSD; the upstream `package.xml` and `README.md` are
  preserved.
- `droid_panda_2f85.xml` is a project modification that removes the stock Panda
  hand, mounts the ARG85/Robotiq gripper at the Panda flange, assigns every
  matching visual geom to MuJoCo group 2, and uses dark gripper materials to
  match the DROID real robot.

The gripper mount used by the experiment is:

```text
Panda link7
  -> translation [0, 0, 0.107] m
  -> quaternion [0.70710678, 0, 0, 0.70710678] (MuJoCo wxyz)
  -> robotiq_85_base_link
```

The experiment sets all six gripper joint positions directly from the DROID
scalar gripper state; no actuator or equality constraint is required for
rendering.
