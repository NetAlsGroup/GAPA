<div align="center">
  <img src="assets/LOGO.png" width="400" alt="GAPA Logo"/>
  <h3>GAPA: Genetic Algorithm Library for Perturbed Substructure Optimization</h3>
  <p>Local Python-first graph optimization with unified workflow, monitoring, extensibility, and accelerated execution.</p>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2412.20980">
    <img src="https://img.shields.io/badge/arxiv-2412.20980-red" alt="arXiv">
  </a>
  <a href="https://pypi.org/project/gapa/">
    <img src="https://img.shields.io/pypi/v/gapa?logo=python" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/gapa/">
    <img src="https://img.shields.io/badge/python-3.9+-orange?logo=python" alt="Python Version">
  </a>
  <img src="https://img.shields.io/github/last-commit/NetAlsGroup/GAPA" alt="GitHub last commit">
  <a href="https://github.com/NetAlsGroup/GAPA">
    <img src="https://img.shields.io/github/stars/NetAlsGroup%2FGAPA" alt="GitHub Stars">
  </a>
</div>

For Chinese documentation, see [README.zh-CN.md](README.zh-CN.md).

## What GAPA Is

GAPA is a local Python library for perturbed substructure structure optimization (PSSO) with genetic algorithms.
It focuses on a unified user-facing workflow:

- load a registered dataset with `DataLoader`
- run an algorithm through `Workflow`
- inspect progress and outputs with `Monitor`
- access resource planning through `ResourceManager`
- extend behavior with custom `Algorithm` subclasses

The base service includes local parallel acceleration for genetic algorithms. Web, service, and heterogeneous distributed execution remain available as advanced services.

## Installation

Recommended package install (to be updated):

```bash
pip install gapa
```

Install from source:

```bash
git clone https://github.com/NetAlsGroup/GAPA.git
cd GAPA
pip install -e .
```

Optional dependency tiers:

```bash
pip install "gapa[full]"
pip install "gapa[attack]"
pip install "gapa[distributed]"
pip install "gapa[dev]"
```

## Supported Tasks

- `CND`: critical node detection
- `CDA`: community detection attack
- `NCA`: node classification attack
- `LPA`: link prediction attack

## Algorithm Example

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
    verbose=False,
)
workflow.run(steps=20)

print(monitor.result())
```

## Quick Validation

Installed package validation:

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
    verbose=False,
)
workflow.run(steps=5)

print(monitor.result())
```

Run with example scripts:

```bash
python examples/api/workflow.py
python examples/algorithms/CND/sixdst.py
```

Repository examples are source-tree examples. Run them after `pip install -e .`.

More runnable examples:

- [examples/api/workflow.py](examples/api/workflow.py)
- [examples/api/monitor.py](examples/api/monitor.py)
- [examples/README.md](examples/README.md)

## Core API

- `DataLoader`: load a registered dataset with `DataLoader.load(name)`
- `Workflow`: run, single-step, pause, resume, and reset algorithm execution
- `Monitor`: inspect `status()`, `result()`, `history()`, and `report()`
- `ResourceManager`: query resources and request strategy plans
- `Algorithm`: define custom algorithm objects that plug into `Workflow`

## Extensibility

GAPA keeps the execution path unified for built-in and user-defined algorithms. A user-defined algorithm object can be passed into the same `Workflow` used by built-in algorithms.

Minimal API examples:

- [data_loader.py](examples/api/data_loader.py)
- [workflow.py](examples/api/workflow.py)
- [monitor.py](examples/api/monitor.py)
- [resource_manager.py](examples/api/resource_manager.py)
- [algorithm.py](examples/api/algorithm.py)

## Advanced Usage

- docs hub: [docs/README.md](docs/README.md)
- advanced usage: [docs/ADVANCED_USAGE.md](docs/ADVANCED_USAGE.md)
- operations and QA: [docs/OPERATIONS_AND_QA.md](docs/OPERATIONS_AND_QA.md)
- remote runtime: `server/`, `app.py`, `server_agent.py`
- web console: `web/`

## Implemented Algorithms

| Abbr             | Years        | Type        | Ref                           | Code                                                                                                                                   |
|------------------|--------------|-------------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| <center>Q-Attack | <center>2019 | <center>CDA | <center>[\[1\]](#r1)</center> | <center>-                                                                                                                              |
| <center>CGN      | <center>2022 | <center>CDA | <center>[\[2\]](#r2)</center> | <center>[Link](https://github.com/HNU-CCIE-AI-Lab/CGN)                                                                                 |
| <center>CDA-EDA  | <center>2020 | <center>CDA | <center>[\[3\]](#r3)</center> | <center>-                                                                                                                              |
| <center>CutOff   | <center>2023 | <center>CND | <center>[\[4\]](#r4)</center> | <center>-                                                                                                                              |
| <center>SixDST   | <center>2024 | <center>CND | <center>-                     | <center>-                                                                                                                              |
| <center>TDE      | <center>2022 | <center>CND | <center>[\[5\]](#r5)</center> | <center>-                                                                                                                              |
| <center>LPA-EDA  | <center>2019 | <center>LPA | <center>[\[6\]](#r6)</center> | <center>[Link](https://github.com/Zhaominghao1314/Target-Defense-Against-Link-Prediction-Based-Attacks-via-Evolutionary-Perturbations) |
| <center>LPA-GA   | <center>2019 | <center>LPA | <center>[\[6\]](#r6)</center> | <center>-                                                                                                                              |
| <center>GANI     | <center>2023 | <center>NCA | <center>[\[7\]](#r7)</center> | <center>[Link](https://github.com/alexfanjn/GANI)                                                                                      |
| <center>NCA-GA   | <center>2018 | <center>NCA | <center>[\[8\]](#r8)</center> | <center>[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack?tab=readme-ov-file)                                              |

## Citation

If you use GAPA in your research, cite:

```bibtex
@article{
    title = Efficient Parallel Genetic Algorithm for Perturbed Substructure Optimization in Complex Network
    author = Shanqing Yu, Meng Zhou, Jintao Zhou, Minghao Zhao, Yidan Song, Yao Lu, Zeyu Wang, Qi Xuan
    journal = arXiv preprint arXiv:2412.20980
    year = {2024}
    doi = {https://doi.org/10.48550/arXiv.2412.20980}
}
```

## References

<p id="r1">
[1] Jinyin Chen, Lihong Chen, Yixian Chen, Minghao Zhao, Shanqing Yu,
Qi Xuan, and Xiaoniu Yang. Ga-based q-attack on community detection.
IEEE Transactions on Computational Social Systems, 6(3):491–503, 2019.
</p>

<p id="r2">
[2] Liu Dong, Zhengchao Chang, Guoliang Yang, and Enhong Chen.
"Hiding ourselves from community detection through genetic algorithms." Information Sciences 614 (2022): 123-137.
</p>

<p id="r3">
[3] Shanqing Yu, Jun Zheng, Jinyin Chen, Qi Xuan, and Qingpeng Zhang.
Unsupervised euclidean distance attack on network embedding. In 2020 IEEE Fifth International Conference on Data Science in Cyberspace
(DSC), pages 71–77, 2020.
</p>

<p id="r4">
[4] Yu, Shanqing, Jiaxiang Li, Xu Fang, Yongqi Wang, Jinhuan Wang, Qi Xuan, and Chenbo Fu.
"GA-Based Multipopulation Synergistic Gene Screening Strategy on Critical Nodes Detection." IEEE Transactions on Computational Social Systems (2023).
</p>

<p id="r5">
[5] Yu, Shanqing, Yongqi Wang, Jiaxiang Li, Xu Fang, Jinyin Chen, Ziwan Zheng, and Chenbo Fu.
"An improved differential evolution framework using network topology information for critical nodes detection." IEEE Transactions on Computational Social Systems 10, no. 2 (2022): 448-457.
</p>

<p id="r6">
[6] Yu, Shanqing, Minghao Zhao, Chenbo Fu, Jun Zheng, Huimin Huang, Xincheng Shu, Qi Xuan, and Guanrong Chen.
"Target defense against link-prediction-based attacks via evolutionary perturbations." IEEE Transactions on Knowledge and Data Engineering 33, no. 2 (2019): 754-767.
</p>

<p id="r7">
[7] Fang, Junyuan, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, and K. Tse Chi.
"Gani: Global attacks on graph neural networks via imperceptible node injections." IEEE Transactions on Computational Social Systems (2024).
</p>

<p id="r8">
[8] Dai, Hanjun, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song.
"Adversarial attack on graph structured data." In International conference on machine learning, pp. 1115-1124. PMLR, 2018.
</p>
