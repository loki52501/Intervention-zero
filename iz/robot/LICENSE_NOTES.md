# Unitree G1 model licence notes

- Model: Unitree G1 (MJCF), from MuJoCo Menagerie, directory `unitree_g1`.
- Upstream: https://github.com/google-deepmind/mujoco_menagerie (subfolder `unitree_g1`).
  The README says it is derived from Unitree's publicly available MJCF
  (https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml).
- Licence: BSD-3-Clause (README: "This model is released under a BSD-3-Clause License"; text in `unitree_g1/LICENSE`).
- Copyright: (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"). All rights reserved.
- The clone lives in `assets/menagerie/` and is gitignored (sparse checkout of `unitree_g1` only).
  `iz/robot/g1.py` reads `g1.xml` at runtime and emits a modified plain-XML copy; the LICENSE
  text must be kept with any redistribution of the mesh files or the derived XML.
