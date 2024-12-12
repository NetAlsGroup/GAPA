GAPA is a Python library that accelerates Perturbed SubStructure Optimization(PSSO).
This challenging field has many vital applications, such as Community Detection Attacks (CDA), Critical Node Detection(CND), Node Classification Attacks (NCA), and Link Prediction Attacks (LPA).
<br>
<br>
GAPA proposes a parallel acceleration framework to achieve fast computation of the Genetic Algorithm (GA) in PSSO.
Currently, GAPA contains 10 PSSO algorithms implemented by GA.
All algorithms built upon [PyTorch](https://github.com/pytorch/pytorch).
<br><br>

<h3>
Basic Environment
</h3>

* `python == 3.9`
* `pytorch == 2.3.0`

See `requirements.txt` for more information.
<br>

<h3>
Installation
</h3>
Install from pip

```
pip install gapa
```

<br>
<h3>
Install from source
</h3>

```
git clone xxx
cd GAPA
python setup.py install
```

If you find the dependencies are complex to install, please try the following: `python setup_empty.py install` (only install GAPA without installing other packages)


<br>
<h3>
File tree

</h3>

```
├─GAPA
| |
| |-gapa  # Origin Files with Framework and Algorithms
| | └─algorithm  # The Files of Algorithms
| | | ├─CDA
| | | ├─CND
| | | ├─LPA
| | | └─NCA
| | ├─DeepLearning  # Some Key File for NCA Task
| | └─framework
| | | ├─body.py  # The Main Part of GA like Mutation
| | | ├─controller.py  # The Workflow Controller of all Acceleration Modes
| | | └─evaluator.py  # Class of Fitness Function
| | └─utils  # Some helpful Functions
| |-tests  # Examples for ten Algorithms and Custom Method
| | ├─absolute_path.py  # The main path of This project and Others.
| | ├─CDA_new.py  #  Kickstart Algorithms of CDA Tasks
| | ├─CND_new.py  #  Kickstart Algorithms of CND Tasks
| | ├─LPA_new.py  #  Kickstart Algorithms of LPA Tasks
| | ├─NCA_new.py  #  Kickstart Algorithms of NCA Tasks
| | ├─Custom.py  #  Example for Custom Method
| | ├─run.py  #  Kickstart a Custom Method
```

<br>
<h3>
Test Examples
</h3>

We provide four example files in `tests/` to illustrate how to use our optimization code. The relationship between the algorithm and the reference file is as follows:

| File               | Algorithm             |
|--------------------|-----------------------|
| <center>CDA_new.py | QAttack, CGN, CDA-EDA |
| <center>CND_new.py | CutOff, SixDST, TDE   |
| <center>NCA_new.py | GANI, NCA-GA          |
| <center>LPA_new.py | LPA-EDA, LPA-GA       |

<br>
Kickstart an algorithm:

```
python CND_new.py --dataset=ForestFire_n500 --method=SixDST --pop_size=100 --mode=m
```
or you can choose your way to start an algorithm.
<br>

<h3>
Custom Method
</h3>

We also provide a way to start a custom method. See more details in `custom.py`,  `run.py` and `absolute_path.py`.
<br>

1. Set up your path.

```
project_path = ...  # ~/
dataset_path = ...
# If NCA -> Set model path
model_path = ...
```


2. Create your own evaluator and controller classes.

```
import torch
from gapa.framework.evaluator import BasicEvaluator
from gapa.framework.controller import CustomController


class ExampleEvaluator(BasicEvaluator):
    def __init__(self, pop_size, adj: torch.Tensor, device):
        super().__init__(
            pop_size=pop_size,
            device=device
        )
        ...
    def forward(self, population):
        ...
        

class ExampleController(CustomController):
    def __init__(self, budget, pop_size, pc, pm, side, mode, num_to_eval, device):
        super().__init__(
            side=side,
            mode=mode,
            budget=budget,
            pop_size=pop_size,
            num_to_eval=num_to_eval,
            device=device
        )
        self.crossover_rate = pc
        self.mutate_rate = pm
        ...
        
    def setup(self, data_loader, evaluator, **kwargs):
        ...
        return evaluator

    def init(self, body):
        ...
        return ONE, population

    def SelectionAndCrossover(self, body, population, fitness_list, ONE):
        ...
        return crossover_population

    def Mutation(self, body, crossover_population, ONE):
        ...
        return mutation_population

    def Eval(self, generation, population, fitness_list, critical_genes):
        metric = MetricFunc(args)
        return {
            "Metric": metric
        }
```

3. Create a new python file and initialize your device. Use world_size to choose the desired number of processes:

```
import os
from gapa.utils.init_device import init_device
device, world_size = init_device(world_size=2)
```
4. Then import other package.

```
import torch
import networkx as nx
from custom import ExampleEvaluator, ExampleController
from absolute_path import dataset_path
from gapa.utils.DataLoader import Loader
from gapa.framework.controller import Start
from gapa.framework.body import Body
```

5. Load your data.

```
dataset = ...
data_loader = Loader(
    dataset, device
)
...
```

6. Run your method.

```
body = Body(nodes_num, budget, pop_size, "min", device)
evaluator = ExampleEvaluator(pop_size, adj, device)
controller = ExampleController(budget, pop_size, pc, pm, fit_side, "s", num_to_eval, device)
Start(max_generation, data_loader, controller, evaluator, body, world_size)
```


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

<h3>
Citing GAPA
</h3>
If you use GAPA in your research and want to cite in your work, please use:
<br>

```
@article{
    title = xxx
    author = xxx
    journal = xxx
    year = 2024
    doi = {xxx}
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
