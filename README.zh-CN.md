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

常用环境变量：

- `GAPA_MNM_MAX_WORKERS`（默认 `4`）
- `GAPA_MNM_REFRESH_S`（默认 `2.0`）
- `GAPA_MNM_MIN_CHUNK_SIZE`（默认 `6`，合并过小分片，减少通信主导调度）
- `GAPA_MNM_COMM_WINDOW_ITERS`（默认 `3`，comm-guard 按滑动窗口评估 overhead/compute）
- `GAPA_MNM_FP16_MIN_ROWS`（默认 `24`，仅大分片使用 FP16 传输）
- `GAPA_RPC_COMPRESS_MIN_BYTES`（默认 `2048`，低于阈值不压缩，仅走带版本头帧）
- `GAPA_RPC_COMPRESS_MIN_SAVING`（默认 `0.05`，压缩收益不足则保留原始帧）
- `GAPA_RPC_TORCH_LEGACY_SERIALIZATION`（默认 `1`，RPC 维持 legacy torch 序列化路径）

性能基线与回归门禁：
- 生成性能基线（默认 synthetic）：`python examples/run_perf_baseline.py --profile small --source synthetic`。
- 基于最小真实运行链路生成基线（live）：`python examples/run_perf_baseline.py --profile small --source live --live-samples 48`。
- 生成发布级基线（含真实算法路径采样）：`python examples/run_perf_baseline.py --profile release_small --source real --real-dataset ForestFire_n500 --real-generations 1 --real-pop-size 12 --real-runs 2`。
- 重新采集当前指标并执行门禁对比：
  - `python tests/perf_regression_gate.py --baseline <baseline.json> --current <current.json> --output <gate.json>`
- 门禁覆盖 mode 集合一致性（缺失/额外 mode 必须失败）以及 `S/SM/M/MNM` 的吞吐下降、时延上升、恢复时延上升和远程失败率漂移。
- 基线输出新增可追溯字段：`host_facts`、`config_snapshot`，以及 `source=real` 时的 `real_workload_meta`。

长稳与混沌稳定性验证：
- 执行确定性 soak+chaos 验证：`python tests/soak_chaos_stability.py --iterations 80 --output .multi-agents/qa/qa-soak-and-chaos-stability-hardening-iteration-14.json`。
- 发布前继续执行跨平台 P0 门禁：`python .multi-agents/scripts/run_cross_platform_mode_gate.py`。

MNM 通信优化验证（iteration-16）：
- 生成通信前后对比报告：`python tests/mnm_comm_iteration16_report.py --output .multi-agents/qa/mnm-communication-optimization-iteration-16.json`。
- QA 模板结果：`.multi-agents/qa/qa-mnm-communication-algorithm-optimization-iteration-16.json`。

发布候选（RC）交付包：
- RC 清单、发布说明、回滚手册和推广候选集中在 `docs/` 目录。
- 建议从 `docs/RC_CHECKLIST_ITERATION_15.md` 开始，并结合 `.multi-agents/qa/qa-release-candidate-closure-and-promotion-iteration-15.json` 校验证据。

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
