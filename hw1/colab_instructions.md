## 在 Colab 上运行作业
如果你在安装时遇到问题或没有计算资源，请按以下步骤在 Colab 上运行作业。

1. 将项目文件夹上传到 Google Drive
2. 打开 [https://colab.research.google.com](https://colab.research.google.com/) 并创建一个新笔记本
3. 在新的单元格中运行
	```
	from google.colab import drive
	drive.mount('/content/gdrive')
	```
4. 输入你的账号信息
5. 使用终端命令（cd 和 ls）切换到你的文件夹位置
6. 安装依赖
	```
	!pip install -r requirements.txt
	```

	（有些依赖可能已预装，例如如果覆盖 ipython 会导致 Colab 出问题，则无需安装它）
7. 安装模块
	```
	!pip install -e .
	```
8. 在左侧找到名为 “Files” 的标签并进入你的项目目录。
9. 点击 Python 文件进行编辑。
10. 运行代码的示例脚本如下：
	```
	!python cs224r/scripts/run_hw1.py \
		--expert_policy_file cs224r/policies/experts/Ant.pkl \
		--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
		--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
		--video_log_freq -1
	```
