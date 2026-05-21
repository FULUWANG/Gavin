# Vibe Coding Data Science Workspace

信用、借贷、风控和数据分析工作的本地工作区。

## 推荐启动方式

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
MPLCONFIGDIR=.cache/matplotlib \
JUPYTER_CONFIG_DIR=.jupyter/config \
JUPYTER_DATA_DIR=.jupyter/data \
JUPYTER_RUNTIME_DIR=.jupyter/runtime \
IPYTHONDIR=.ipython \
jupyter lab
```

也可以用项目脚本启动：

```bash
scripts/start_jupyter.sh
```

## 可选增强

`requirements-optional.txt` 里放了可能需要系统依赖的增强包。比如 Apple Silicon 上的 XGBoost 通常需要先安装 OpenMP runtime：

```bash
brew install libomp
python -m pip install -r requirements-optional.txt
```

## 目录结构

- `data/raw/`: 原始数据，只读保存。
- `data/interim/`: 中间处理结果。
- `data/processed/`: 建模或分析用的干净数据。
- `notebooks/`: 探索分析、建模实验、汇报图表。
- `src/`: 可复用的数据处理、特征工程、建模代码。
- `reports/`: 输出图、表、报告草稿。

## 风控/信贷分析常用方向

- 贷前：申请评分、准入策略、收入/负债/欺诈风险分析。
- 贷中：额度管理、行为评分、预警规则、滚动率分析。
- 贷后：催收分层、回收率、迁徙矩阵、Vintage 分析。
- 模型：Logistic Regression、GBDT/XGBoost、校准、KS/AUC、PSI、分箱、WOE/IV。
