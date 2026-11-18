# 工具目录

- `migration_preflight.sh`：检查 ROS 包、命令、关键文件、launch XML、SBPL 架构
  和设备权限；
- `create_migration_archive.sh`：生成精简可运行 ZIP、执行完整性测试并输出
  SHA-256；
- `build_sbpl_runtime.sh`：按锁定 commit 为当前主机重编 signed-char SBPL；
- `build_arm64_sbpl.sh`：保留的历史兼容入口，转调上述通用脚本；
- `generate_scaled_mprim.py`：生成指定分辨率的动作原语；
- `rebase_rectified_profile.py`、`transfer_camera_profile.py`：相机配置迁移辅助。

生成迁移包：

```bash
./tools/create_migration_archive.sh /path/to/output.zip
```

脚本不会覆盖已有 ZIP，也不会删除原工作区中的任何实验数据。
