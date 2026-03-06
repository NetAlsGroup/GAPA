# GAPA 中文文档

本文件为中文说明入口。主文档请优先参考英文版 [README.md](README.md)。

## 项目定位

GAPA 是面向 PSSO 的遗传算法加速库，支持：

- 本地单进程（`s`）
- 单机多卡（`sm`）
- 本地分布式多进程（`m`）
- 异构多机分布式适应度加速（`mnm`）

## 安装分层

推荐新手安装：

```bash
pip install gapa
```

源码可编辑安装：

```bash
git clone https://github.com/NetAlsGroup/GAPA.git
cd GAPA
pip install -e .
```

可选依赖层：

```bash
pip install "gapa[full]"         # 内置算法、绘图与研究依赖
pip install "gapa[attack]"       # 攻击类依赖
pip install "gapa[distributed]"  # app/server/远程运行依赖
pip install "gapa[dev]"          # 开发工具链 + 全量可选依赖
```

源码依赖文件也按同样层级拆分：

```bash
pip install -r requirements.txt
pip install -r requirements/full.txt
pip install -r requirements/attack.txt
pip install -r requirements/distributed.txt
pip install -r requirements/dev.txt
```

`requirements.txt` 现在只表示轻量 core 安装。

## 快速开始

官方首跑命令：

```bash
python -m gapa demo
```

安装后的快捷命令：

```bash
gapa demo
```

这个 quickstart 会：

- 运行包内置的小图 demo
- 把摘要结果写到 `results/quickstart/runs/`
- 不依赖 `examples/` 或仓库数据集

你会看到：

- `requested` / `resolved` mode
- best fitness 摘要
- summary report 路径

验证环境与 smoke 路径：

```bash
python -m gapa doctor
```

下一步：

1. 打开进阶文档导航： [docs/README.zh-CN.md](docs/README.zh-CN.md)
2. 本地脚本与自定义算法： [docs/ADVANCED_USAGE.zh-CN.md](docs/ADVANCED_USAGE.zh-CN.md)
3. 性能、稳定性、QA 与发布： [docs/OPERATIONS_AND_QA.zh-CN.md](docs/OPERATIONS_AND_QA.zh-CN.md)

## 项目表面

- `gapa/`：核心包、workflow API、内置 demo 入口
- `examples/`：进阶脚本入口
- `server/`、`app.py`、`server_agent.py`：服务与远程运行时
- `docs/`：第二层文档入口，承接进阶使用、运维、QA 与发布资料

## 更新日志

- 详细更新历史请查看 [CHANGELOG.txt](CHANGELOG.txt)。

## 进阶资料

- 文档导航： [docs/README.zh-CN.md](docs/README.zh-CN.md)
- 进阶使用： [docs/ADVANCED_USAGE.zh-CN.md](docs/ADVANCED_USAGE.zh-CN.md)
- 运维与 QA： [docs/OPERATIONS_AND_QA.zh-CN.md](docs/OPERATIONS_AND_QA.zh-CN.md)
