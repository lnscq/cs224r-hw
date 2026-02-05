# CS224R HW1: Behavior Cloning (BC) 实验报告

> **环境**：`Ant-v4`  
> **方法**：行为克隆（Behavioral Cloning）  
> **结果**：成功复现专家策略，评估回报 ≈ 4662（接近专家水平）

---

## 📸 演示视频

![](img\imageData.gif)

---

## ✅ 实验成果

- **成功运行 BC 算法**，在 `Ant-v4` 环境上达到 **4662.85 的平均回报**
- 专家策略回报通常为 4500~5000，说明 BC 几乎完美模仿了专家
- 训练过程稳定，无崩溃，日志完整
- 支持 TensorBoard 可视化（`.tfevents` 文件）

### 关键指标（Iteration 0）
| 指标                  | 值                                |
| --------------------- | --------------------------------- |
| `Eval_AverageReturn`  | 4662.85                           |
| `Eval_StdReturn`      | 0.0                               |
| `Eval_AverageEpLen`   | 1000.0（满长度）                  |
| `Train_AverageReturn` | 4713.65                           |
| `Training Loss`       | -15.75（负值因使用高斯 NLL loss） |

---

## 🧠 学到的核心内容

### 1. **行为克隆（Behavioral Cloning）原理**
- 将强化学习问题转化为**监督学习**：输入状态 `s`，预测专家动作 `a`
- 使用 **MSE 或负对数似然（NLL）** 作为损失函数
- 依赖高质量的专家轨迹数据集

### 2. **模块化 RL 架构设计**
- **Agent**：高层策略管理器（`BCAgent`）
- **Policy**：底层神经网络（`MLPPolicySL`）
- **Replay Buffer**：存储和采样轨迹
- **Trainer**：控制训练循环（`BCTrainer`）
- **Logger**：记录指标与视频

### 3. **PyTorch + NumPy 兼容性管理**
- 理解了 **NumPy 1.x vs 2.x** 的兼容性问题
- 掌握了 `torch.from_numpy()` 与设备（CPU/GPU）迁移

### 4. **Gym 环境交互**
- 熟悉 `reset()`, `step()`, `render()` 接口
- 理解 `obs`, `action`, `reward`, `done` 数据流

---

## ⚠️ 踩过的坑与解决方案

### 🔴 **坑 1：NumPy 2.x 与 PyTorch 不兼容**
```text
RuntimeError: Numpy is not available
```
- **原因**：PyTorch 尚未支持 NumPy 2.0+
- **解决**：
  ```bash
  pip install "numpy<2"  # 降级到 1.26.x
  ```

### 🔴 **坑 2：ReplayBuffer 返回 tuple，但 trainer 当 dict 用**
```text
TypeError: tuple indices must be integers or slices, not str
```
- **原因**：`sample_random_data()` 返回 `(obs, acs, ...)`，但代码写成 `batch['obs']`
- **解决**：修改 `bc_trainer.py`：
  ```python
  # 错误
  ob_batch, ac_batch = batch['obs'], batch['acts']
  # 正确
  ob_batch, ac_batch = batch[0], batch[1]
  ```

### 🔴 **坑 3：调用不存在的 `agent.update()`**
```text
AttributeError: 'BCAgent' object has no attribute 'update'
```
- **原因**：`BCAgent` 的训练接口是 `train()`，不是 `update()`
- **解决**：修改 `bc_trainer.py`：
  ```python
  # 错误
  self.agent.update(ob, ac)
  # 正确
  self.agent.train(ob, ac)
  ```
---

## 📁 项目结构说明

```
hw1/
├── cs224r/
│   ├── agents/          # BCAgent
│   ├── policies/        # MLPPolicySL, LoadedGaussianPolicy
│   ├── infrastructure/  # ReplayBuffer, BCTrainer, logger, pytorch_util
│   └── scripts/         # run_hw1.py
├── data/                # 输出目录（logs, videos, tfevents）
├── test_video.py        # 视频保存测试脚本
└── README.md            # 本文件
```

---

## 📌 总结

本次作业深入理解了**模仿学习**的基本思想，并通过实践掌握了 RL 系统的模块化设计。虽然遇到了环境兼容性、接口不一致、视频渲染等挑战，但通过系统性排查，最终成功复现了高性能的行为克隆 agent。


---
