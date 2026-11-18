# 2026-08-12 已跑通的 Stage 1 + Stage 2 比赛基线

## 结论

截至 2026-08-12，工程中已经过连续完整实车验证、应当保留为回退基线的方案是：

```text
competition_stage12_auto.py
  ├─ Stage 1
  │    RGB 黄线/双圆片冻结 + 起点雷达入口目标
  │    GlobalPlanner + compact-obstacle TEB
  │    每枚圆片只封闭较窄的一侧通道
  │
  └─ Stage 2
       起点雷达冻结三面墙和终点
       空白地图 + 冻结激光场景 + AMCL
       GlobalPlanner + TEB
```

该组合是当前“已证明能跑通”的正式基线。后续算法实验不得覆盖或静默改变它的默认插件、单侧通道封闭逻辑和启动入口。

## 两次连续完整成功记录

| 实车轮次 | Stage 1 | Stage 1 导航 | Stage 1 目标 | Stage 2 | Stage 2 导航 | Stage 2 目标 |
|---|---:|---:|---|---:|---:|---|
| Run A | 51.69 s | 31.59 s | `(2.318,-0.307,-0.035 rad)` | 41.94 s | 29.89 s | `(2.089,-0.013,0.039 rad)` |
| Run B | 53.58 s | 33.00 s | `(2.242,-0.291,-0.018 rad)` | 43.90 s | 31.49 s | `(2.081,-0.002,0.034 rad)` |

两轮均由各阶段 runner 返回 `SUCCEEDED`，Stage 1 与 Stage 2 使用相互隔离的 ROS 会话，前一阶段完全关闭后才启动后一阶段。

原始 runner 摘要已固化在：

- `docs/trials/stage12_validated_run_summaries_20260812.json`

## Stage 1 已验证拓扑

```text
视觉 ACQUIRE
  → 黄线走廊锁定
  → 两枚圆片确认并冻结到 odom/costmap
  → 视觉 LOCKED
  → 雷达冻结窄道入口前目标
  → 单侧 passage closure 选择较宽绕行拓扑
  → GlobalPlanner
  → compact-obstacle TEB
  → move_base 唯一发布 /cmd_vel
```

单侧封闭规则一次最多封闭每枚圆片的一侧窄缝，不设置“两侧都不满足就停车”的分支；该规则负责选择全局绕行拓扑，TEB负责具体全向轨迹。

## SBPL 实验的边界

新增的 `competition_stage1_sbpl_auto.launch` 是独立、默认锁车的候选方案：

```text
SBPL XYTheta Lattice + compact-obstacle TEB
passage closures = false
```

其目的仅是验证：不进行人工单侧封闭时，SBPL能否利用 `x/y/yaw` 状态、`0.34 × 0.33 m`矩形 footprint和全向运动基元，自主选择真实可通行的一侧。

在完成实车验证前，SBPL候选不能被称为新的正式最优方案，也不能替换本文件记录的已跑通基线。
