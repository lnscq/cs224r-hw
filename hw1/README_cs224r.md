## 环境配置

请参阅 [installation.md](installation.md) 获取安装说明。如果你想使用 Google Colab，请参阅 [colab_instructions.md](colab_instructions.md)。

## 完成代码

填写标有 `TODO` 的部分。重点关注：
 - [infrastructure/bc_trainer.py](cs224r/infrastructure/bc_trainer.py)
 - [policies/MLP_policy.py](cs224r/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs224r/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs224r/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs224r/infrastructure/pytorch_util.py)

查找标有 `HW1` 的部分，以了解你的改动将如何被使用。
你可能还会用到以下文件：
 - [scripts/run_hw1.py](cs224r/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs224r/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](cs224r/agents/bc_agent.py)

更多细节请参阅作业 PDF。

### 常见问题：

 - **我们应该使用什么损失函数 / 采样策略？** 你可以使用任何行为克隆策略，只要你的策略表现符合作业要求。你也可以在架构决策部分对不同策略进行比较！注意在 PyTorch 中，从随机策略采样有两种方式：`rsample()` 可以反向传播梯度，而 `sample()` 不能，因此训练时你可能更适合使用前者。如果你想了解原因，可以阅读 [重参数化技巧](https://gregorygundersen.com/blog/2018/04/29/reparameterization/)。
 - **需要处理离散动作空间吗？** 我们使用的所有仿真环境都输出连续动作。

## 运行代码

提示：调试时建议保留参数 `--video_log_freq -1`，它会禁用视频记录并加速实验。当然，你也可以移除它来保存你的超棒策略视频！

### 第 1 部分（行为克隆）
题目 1 的命令：

```
python cs224r/scripts/run_hw1.py \
	--expert_policy_file cs224r/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

务必再尝试另一个环境。
更多需要运行的内容请参阅作业 PDF。
如需生成策略视频，请移除 `--video_log_freq -1` 参数。

### 第 2 部分（DAgger）
第 1 题命令：
（注意 `--do_dagger` 参数，以及更大的 `n_iter` 值）

```
python cs224r/scripts/run_hw1.py \
    --expert_policy_file cs224r/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1
```

务必再尝试另一个环境。
更多需要运行的内容请参阅作业 PDF。

## 可视化保存的 TensorBoard 事件文件：

你可以使用 tensorboard 可视化运行结果：
```
tensorboard --logdir data
```

你将看到标量汇总信息以及训练后策略的视频（在 “images” 标签中）。

你也可以用逗号分隔的列表选择可视化指定的运行：
```
tensorboard --logdir data/run1,data/run2,data/run3...
```
