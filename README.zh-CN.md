# GAPA 中文文档

本文件为中文说明入口。主文档请优先参考英文版 [README.md](README.md)。

## 项目定位

GAPA 是面向 PSSO 的遗传算法加速库，支持：

- 本地单进程（`s`）
- 单机多卡（`sm`）
- 本地分布式多进程（`m`）
- 异构多机分布式适应度加速（`mnm`）

## 快速开始

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode s
```

远程单服务器模式（`s/sm/m`）：

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --use-strategy-plan
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --no-strategy-plan
```

异构多机 MNM：

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode mnm --servers "Server 6"
```

## 服务启动

本地编排服务：

```bash
python app.py
```

远程 agent：

```bash
uvicorn server_agent:app --host 0.0.0.0 --port 7777
```

## 配置文件

- `servers.json`：远程服务器列表与地址
- `algorithms.json`：通用/自定义算法统一注册入口

## 项目结构

```text
GAPA/
├── app.py                    # 本地编排服务（API + Web）
├── server_agent.py           # 远程计算 agent 服务
├── algorithms.json           # 通用/自定义算法注册
├── servers.json              # 远程服务器配置
├── gapa/                     # 核心框架与内置算法
├── server/                   # 运行时模块（锁、队列、分布式评估）
├── autoadapt/                # StrategyPlan 与自适应路由
├── examples/                 # 最简脚本示例
├── tests/                    # 回归测试
├── dataset/                  # 内置数据集
└── web/                      # 前端静态资源
```

## 更新日志

- 详细更新历史请查看 [CHANGELOG.txt](CHANGELOG.txt)。

## 脚本示例索引

- `examples/run_sixdst.py`：主运行脚本
- `examples/resource_scheduler.py`：资源查看/锁定/策略规划
- `examples/run_lock_keepalive.py`：锁续期示例
- `examples/run_analysis_queue.py`：任务队列示例
- `examples/run_report_export.py`：监控导出示例
- `examples/run_trends.py`：趋势统计示例
