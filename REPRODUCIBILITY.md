# 复现说明

仓库包含源码、配置、结果摘要和图表。原实验环境为 Ubuntu 24.04、Python 3.12 和两张 24 GB RTX 4090。

## 生成首页结果图

直接读取仓库中的结果摘要，无需模型或显卡：

```bash
python -m pip install matplotlib==3.10.1
python scripts/generate_readme_figure.py
```

需安装微软雅黑、Noto Sans CJK SC 或其他脚本支持的中文字体。图片和矢量 PDF 保存在 `reports/figures_readme/`。

## 运行基础测试

在 Linux 环境中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pytest==8.1.1 rank-bm25==0.2.2 math-verify==0.8.0 sympy==1.13.1
python -m pip install -e . --no-deps
python -m pytest -q
```

测试覆盖协议解析、工具执行、答案验证和奖励计算，无需加载语言模型。部分工具测试依赖 Linux 资源限制；GitHub 的自动检查使用 Linux。

## 训练与评测

先准备可用的 PyTorch 和 CUDA 环境，再安装项目依赖：

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.lock.txt
python -m pip install -e . --no-deps
```

按[资源清单](data/manifests/source_manifest.json)下载指定版本的模型与数据，再结合[阶段记录](PROGRESS.md)、[配置](configs/)和[脚本](scripts/)运行实验。原始软硬件信息见[环境报告](reports/environment_report.md)。

模型权重、适配器、完整数据、预测文件和训练日志未包含在仓库中。准备好相应预测文件和日志后，可运行 `make reproduce-final` 重新计算最终报告；该命令不执行训练或模型推理。

其余已有图表可用 `make reproduce-paper` 重新生成，需要安装 `numpy==1.26.4` 和 `matplotlib==3.10.1`。
