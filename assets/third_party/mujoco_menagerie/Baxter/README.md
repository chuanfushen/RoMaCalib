# Baxter model assets

The robot body and end-effector assets come from
[`RethinkRobotics/baxter_common`](https://github.com/RethinkRobotics/baxter_common)
at commit `6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e`. See
`LICENSE.baxter_common` for the upstream BSD license.

- `baxter_description/`: official Baxter body URDF/Xacro and meshes.
- `rethink_ee_description/`: official electric/pneumatic gripper Xacro and meshes.
- `baxter_full.urdf`: the official Baxter Xacro expanded with the default left and
  right electric grippers. ROS `package://` mesh URIs are repository-relative, and
  DAE references use their matching upstream STL files for MuJoCo compatibility.

`baxter_full.urdf` enables `discardvisual=false` and `fusestatic=false`. MuJoCo's
URDF importer assigns collision geoms to group 0 and visual geoms to group 1 for
this model. Use `visual_geom_group = 1`, as shown in
`config/dream_eval_baxter.toml`. The Panda MJCF continues to use group 2.
