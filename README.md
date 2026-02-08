<div align="center">
  <img src="assets/LOGO.png" width="400" alt="GAPA Logo"/>

  <h3>
    GAPA: A Parallel Accelerated Framework for Graph Structure Optimization
  </h3>

  <p>
    Efficiently solving PSSO problems (CND, CDA, NCA, LPA) via Unified Genetic Algorithms and Multi-level Parallelism.
  </p>
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

GAPA is a Python library for accelerated Perturbed Substructure Structure Optimization (PSSO),
including CND, CDA, NCA, and LPA workloads. It provides a unified GA execution interface across:

- local single-process (`s`)
- single-node multi-GPU (`sm`)
- local distributed multi-process (`m`)
- heterogeneous multi-node distributed fitness acceleration (`mnm`)

All algorithm implementations are based on [PyTorch](https://github.com/pytorch/pytorch).

For Chinese documentation, see [README.zh-CN.md](README.zh-CN.md).

<h3>Requirements</h3>

- Python `3.9+`
- PyTorch `2.3.0+` (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

<h3>Installation</h3>

Install from PyPI:

```bash
pip install gapa
```

Install from source:

```bash
git clone https://github.com/NetAlsGroup/GAPA.git
cd GAPA
python setup.py install
```

Minimal install (without heavy dependency resolution):

```bash
python setup_empty.py install
```

<h3>Quick Start</h3>

Run the script-first baseline example:

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode s
```

Run remote single-server execution (`s`/`sm`/`m`):

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --use-strategy-plan
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --no-strategy-plan
```

Run heterogeneous MNM mode:

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode mnm --servers "Server 6"
```

<h3>Services</h3>

Start local orchestrator service:

```bash
python app.py
```

Start remote server agent on each compute host:

```bash
uvicorn server_agent:app --host 0.0.0.0 --port 7777
```

Main runtime config files:

- `servers.json`: remote server inventory and endpoints
- `algorithms.json`: registry for generic/user-defined algorithms

Useful environment variables:

- `GAPA_MNM_MAX_WORKERS` (default: `4`)
- `GAPA_MNM_REFRESH_S` (default: `2.0`)

<h3>Project Layout</h3>

```text
GAPA/
├── app.py                    # Local orchestrator service (API + web entry)
├── server_agent.py           # Remote compute agent service
├── algorithms.json           # Generic/user algorithm registry
├── servers.json              # Remote server inventory
├── gapa/                     # Core framework + built-in algorithms
│   ├── algorithm/
│   │   ├── CDA/
│   │   ├── CND/
│   │   ├── LPA/
│   │   └── NCA/
│   ├── framework/
│   ├── utils/
│   └── workflow.py           # Unified workflow + monitor
├── server/                   # Runtime modules
│   ├── distributed_evaluator.py
│   ├── resource_lock.py
│   ├── task_queue.py
│   └── algorithm_registry.py
├── autoadapt/                # StrategyPlan and adaptive routing
├── examples/                 # Minimal script-first demos
├── tests/                    # Regression tests
├── dataset/                  # Built-in datasets
└── web/                      # Frontend assets
```

<h3>Examples</h3>

The `examples/` directory is the primary script API for users.

| File | Purpose |
|------|---------|
| `examples/run_sixdst.py` | End-to-end execution in `s`/`sm`/`m`/`mnm` modes |
| `examples/resource_scheduler.py` | Resource listing, lock, and strategy planning |
| `examples/run_lock_keepalive.py` | Lock/renew/release keepalive flow |
| `examples/run_analysis_queue.py` | Queue-based remote scheduling |
| `examples/run_report_export.py` | Unified monitor export and reporting |
| `examples/run_trends.py` | Run trend aggregation |
| `examples/sixdst_custom.py` | Example custom algorithm wrapper |

<h3>Custom Algorithm Integration</h3>

For user-defined or generic algorithms, use `algorithms.json` as a single registration entry point.

Recommended flow:

1. Implement your algorithm wrapper under `examples/` or your own package.
2. Register the algorithm in `algorithms.json` with:
   - `entry`: import path
   - `init_kwargs`: constructor defaults
   - `capabilities`: supported modes and runtime traits
3. Start runs via `Workflow` / `examples/run_sixdst.py` style scripts.
4. For remote or MNM execution, make sure both local `app.py` and remote `server_agent.py` can import the registered entry.

This approach replaces the old ad-hoc `custom.py/run.py` path and keeps registration centralized.


<br>
<h3>
Implemented Algorithms
</h3>

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
<br>

<h3>Changelog</h3>

See [CHANGELOG.txt](CHANGELOG.txt) for detailed release and milestone updates.

<br>

<h3>
Citing GAPA
</h3>
If you use GAPA in your research and want to cite in your work, please use:
<br>

```
@article{
    title = Efficient Parallel Genetic Algorithm for Perturbed Substructure Optimization in Complex Network
    author = Shanqing Yu, Meng Zhou, Jintao Zhou, Minghao Zhao, Yidan Song, Yao Lu, Zeyu Wang, Qi Xuan
    journal = arXiv preprint arXiv:2412.20980
    year = {2024}
    doi = {https://doi.org/10.48550/arXiv.2412.20980}
}
```

<br>
<h3>References</h3>
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
