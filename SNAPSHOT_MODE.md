# 快照存放位置（CPU / GPU）说明

这套实现在测试时适应（TTA）过程中需要不断"存档 / 读档"：把模型参数和优化器状态复制一份，之后可以回滚。
**存档放在主机内存还是显存，是唯一与实现相关、且会影响墙钟时间的选项。**

默认行为是**主机内存（CPU）**；加一个环境变量即可切到显存（GPU）。

## 一、默认：CPU 快照

不设任何 `MOE_*` 环境变量时，行为与上一个提交一致，只有三处正确性修复（见第四节）。

## 二、切到 GPU 快照

```
MOE_SNAP_MODE=gpu
```

切过去以后：

- 存档（`snapshot_cpu`）把模型状态和优化器状态复制到 `cuda:0`，不再走 PCIe
- 读档（`load_snapshot_cpu`）直接在显存内恢复
- 日志里会看到 `[SNAP] mode=gpu（副本留在 cuda:0，不走 PCIe）`
- 优化器的 `step` 计数器仍然保持在 CPU（这是 PyTorch Adam 的原生行为，见第四节）

`MOE_SNAP_MODE=cpu` 可以显式写回默认值。

## 三、复现"CPU vs GPU 快照"的对照

同一个任务、同一个 seed，只改 `MOE_SNAP_MODE`：

```bash
# CPU 快照
env MOE_SNAP_MODE=cpu python md_eval.py -s moe_dsuta -t md_hard -n snap_cpu \
  --config config/system/moe-conv6.yaml \
  --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml \
  --set-moetta expert_schedule=1-8:9,11-48:4 \
                 normal_layer=51,56,59,64,67,72,75,80,83,88,91,96,99,104,107 \
                 route_granularity=sample \
  --set lb_coef=0.0 --set em_coef=0.5 --seed 666

# GPU 快照（只有这一个变量不同）
env MOE_SNAP_MODE=gpu python md_eval.py -s moe_dsuta -t md_hard -n snap_gpu \
  --config config/system/moe-conv6.yaml \
  --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml \
  --set-moetta expert_schedule=1-8:9,11-48:4 \
                 normal_layer=51,56,59,64,67,72,75,80,83,88,91,96,99,104,107 \
                 route_granularity=sample \
  --set lb_coef=0.0 --set em_coef=0.5 --seed 666
```

DSUTA 基线的同一对照用另一个变量（默认 `gpu`，等于原版行为）：

```bash
env SUTA_SNAP_MODE=cpu python md_eval.py -s dsuta -t md_hard -n dsuta_cpu \
  --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml --seed 666
```

## 四、三处正确性修复

| 位置 | 内容 | 为什么必须 |
|---|---|---|
| `load_snapshot_cpu` | GPU 模式下先把优化器状态深拷贝一次再读档 | `optimizer.load_state_dict` 在设备/dtype 一致时不复制张量，会把存档张量直接挂到优化器状态上；Adam 的原地更新随后会改写存档本身 |
| `_step_keep_cpu` | Adam 的 `step` 计数器保持在 CPU | PyTorch 在 `capturable=False` 时把 `step` 建成 CPU 标量；跟着搬到显存会让读档后的运算路径与原生实现不一致 |
| `_to_cpu` | 单次拷贝（`to("cpu", copy=True)`） | 原来 `to("cpu").clone()` 每个张量拷两遍，值相同但浪费一倍带宽 |

## 五、其余开关（全部默认关闭，与上面这个对照无关）

| 开关 | 默认 | 作用 |
|---|---|---|
| `MOE_SNAP_TRAINABLE_ONLY` | `0` | 存档只保留可训练参数（丢弃冻结的基座参数），减少约 2.2 倍搬运量 |
| `MOE_SNAP_PINNED` | `0` | 存档落在锁页内存，PCIe 传输更快 |
| `MOE_START_RESTORE` | `all` | `shared` 时每次"从 start 重启"只回滚共享权重，让专家累积 |
| `MOE_COMMIT_RESTORE` | `all` | `shared` 时每个提交点的恢复也只回滚共享权重 |
| `MOE_CKSUM` | 空 | 给定文件路径时，每个样本写一行慢模型可训练参数的范数与和，用于逐样本对比两次运行是否同轨迹 |
| `MOE_RESET_AUDIT` | 空 | 给定文件路径时，每次重置后写一行状态快照（shared 与 init 的最大差异、范数、和、残留 Adam、学习率） |

> `MOE_START_RESTORE=shared` 单独使用会让专家权重在整条流上无界漂移，实测量级下 WER 发散（md_hard 约 70%），不要单独用于正式实验。
