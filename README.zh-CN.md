# GAPA 中文文档

主文档英文版见 [README.md](README.md)。

## 这是什么

GAPA 是一个面向扰动子结构优化的本地 Python 遗传算法库。它面向用户的主路径是：

- 用 `DataLoader` 加载注册数据集
- 用 `Workflow` 运行算法
- 用 `Monitor` 查看状态、结果和报告
- 用 `ResourceManager` 查询资源与策略规划
- 用自定义 `Algorithm` 扩展算法行为

基础服务包括本地遗传算法并行加速，另外，Web、服务端和异构分布式属于进阶服务。

## 安装

推荐直接安装包（待更新）：

```bash
pip install gapa
```

从源码安装：

```bash
git clone https://github.com/NetAlsGroup/GAPA.git
cd GAPA
pip install -e .
```

可选依赖层：

```bash
pip install "gapa[full]"
pip install "gapa[attack]"
pip install "gapa[distributed]"
pip install "gapa[dev]"
```

## 支持任务

- `CND`：关键节点检测
- `CDA`：社区检测攻击
- `NCA`：节点分类攻击
- `LPA`：链路预测攻击

## 算法案例

```python
from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import SixDSTAlgorithm

data = DataLoader.load("Circuit")
monitor = Monitor()
algorithm = SixDSTAlgorithm(pop_size=16)
workflow = Workflow(
    algorithm, 
    data, 
    monitor=monitor, 
    mode="s", 
    verbose=False
)
workflow.run(steps=20)

print(monitor.result())
```

## 快速验证

已安装包的验证方式：

```python
from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import SixDSTAlgorithm

data = DataLoader.load("Circuit")
monitor = Monitor()
workflow = Workflow(
    SixDSTAlgorithm(pop_size=16), 
    data, 
    monitor=monitor, 
    mode="s", 
    verbose=False
)
workflow.run(steps=5)

print(monitor.result())
```

使用样例脚本运行：

```bash
python examples/api/workflow.py
python examples/algorithms/CND/sixdst.py
```

`examples/` 下的示例属于源码树示例，请先执行 `pip install -e .`。

可直接查看的脚本：

- [examples/api/workflow.py](examples/api/workflow.py)
- [examples/api/monitor.py](examples/api/monitor.py)
- [examples/README.md](examples/README.md)

## 核心 API

- `DataLoader`：通过 `DataLoader.load(name)` 加载注册数据集
- `Workflow`：运行、单步调试、暂停、继续与重置流程
- `Monitor`：提供 `status()`、`result()`、`history()`、`report()`
- `ResourceManager`：查询资源并生成策略规划
- `Algorithm`：定义可接入 `Workflow` 的自定义算法对象

## 可扩展性

GAPA 对内置算法和用户自定义算法使用统一工作流。用户定义的算法对象可以和官方算法一样交给 `Workflow` 执行。

最小 API 示例：

- [data_loader.py](examples/api/data_loader.py)
- [workflow.py](examples/api/workflow.py)
- [monitor.py](examples/api/monitor.py)
- [resource_manager.py](examples/api/resource_manager.py)
- [algorithm.py](examples/api/algorithm.py)

## 进阶使用

- 文档导航： [docs/README.zh-CN.md](docs/README.zh-CN.md)
- 进阶使用： [docs/ADVANCED_USAGE.zh-CN.md](docs/ADVANCED_USAGE.zh-CN.md)
- 运维与 QA： [docs/OPERATIONS_AND_QA.zh-CN.md](docs/OPERATIONS_AND_QA.zh-CN.md)
- 远程运行时：`server/`、`app.py`、`server_agent.py`
- Web 控制台：`web/`

## Implemented Algorithms

Implemented Algorithms 表保持英文主文档现状，见 [README.md](README.md#implemented-algorithms)。

## 引用

如果你在研究中使用 GAPA，可使用英文主文档中的 BibTeX： [README.md](README.md#citation)。

## 参考文献

参考文献列表见英文主文档： [README.md](README.md#references)。
