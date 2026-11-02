# Community mobile-robot research

This directory is the independent community-robot Git worktree. Its origin is
`zhaoshibo3134779695/RL_CAR`, and its cloud branch is `cloud-workspace`.
The cloud directory starts its own snapshot history on that branch; the remote's
existing ROS1 main history is retained without checkout or rewriting.
See [Git usage](GIT-GUIDE.md). RL_CAR is public, so pushed files are public.
Local commits require a successful push before they are protected on GitHub.

New maintained code belongs in src/, configs/, scripts/, tests/ and docs/.
The reviewed SCOPE helper sources are retained as historical references; its
upstream repositories, environments, datasets and outputs stay out of this Git.

This project concerns mobile-robot navigation and future paper experiments.
The current research stage is an HRL literature map and baseline selection;
dynamic avoidance, exploration and semantic/topological navigation are candidate
problem families. It is independent of the robot-arm project.

`research/scope_repro/` preserves the historical SCOPE reproduction, environments,
datasets, outputs and export bundle. Preservation does not reactivate that method
or its historical pending tasks. Existing absolute prefixes remain usable through
the legacy workspace alias.

The separately documented ROS 2 Jazzy/Gazebo workspace on another device has not
been copied here. New experiments should record code commit, environment lock,
input identity and outputs before making paper-performance claims.

The project decision owner remains WorkVault:
`10_Projects/科研/机器人/智慧社区机器人/README.md`.
