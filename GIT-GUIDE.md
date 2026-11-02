# 智慧社区 Git 使用

- 仓库根：`/home/jovyan/work/projects/community-robot`
- GitHub：`https://github.com/zhaoshibo3134779695/RL_CAR`
- 云端分支：`cloud-workspace`，保存当前云端目录的新快照历史。
- 原 main 的 ROS1 工程保持原样，不将其选作当前 HRL/ROS2 基线。

维护自己的代码时放在 src/、configs/、scripts/、tests/ 和 docs/。
research/scope_repro/scripts 中经检查的帮助脚本仅作为历史源码保留；
既有嵌套上游仓库、环境、数据、模型、论文 PDF 和运行结果由 .gitignore 排除。
原文件保持原位；保留历史不代表选定论文方法或执行实验。

RL_CAR 当前是公开仓库，推送的文件公开可见。不要加入账号、设备私有配置、
真实采集资料或尚未决定公开的实验数据。此次没有改变仓库可见性。

日常在 VS Code 源代码管理中选择 Community Robot 仓库，审阅后提交和推送：

```sh
cd /home/jovyan/work/projects/community-robot
git status
git add -- 你实际修改的文件
git diff --cached
git commit -m "说明本次修改"
git push
```

首次推送在云端完成 GitHub 登录后执行：

```sh
gh auth login --hostname github.com --git-protocol https --web
git push -u origin cloud-workspace
```

Git 凭据 helper 仅为这两个项目配置，不改全局 Git。密码/token 不粘贴到
聊天或仓库；登录在浏览器自行授权。提交后还需推送，GitHub 才有该版本。

另一个设备若需要同一云端目录版本，克隆到新目录：

```sh
git clone --branch cloud-workspace https://github.com/zhaoshibo3134779695/RL_CAR.git NEW_DIRECTORY
```

不要自动合并旧 main 与云端快照历史，不用 force push、reset --hard 或 clean。
两个项目各自提交、推送；共享 Isaac/Conda 与另一项目的文件不属于这个仓库。
