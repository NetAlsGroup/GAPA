# 进阶使用

这个文档承接 README 中不适合放在首页新手主路径里的内容，包括脚本入口、服务启动和运行时配置。

## 脚本入口

核心示例位于 `examples/`：

| 文件 | 用途 |
|---|---|
| `examples/run_sixdst.py` | `s` / `sm` / `m` / `mnm` 端到端运行 |
| `examples/resource_scheduler.py` | 资源查看、锁定与策略规划 |
| `examples/run_lock_keepalive.py` | 锁续期 / 释放 keepalive |
| `examples/run_analysis_queue.py` | 队列化远程调度 |
| `examples/run_report_export.py` | 监控导出与结果汇总 |
| `examples/run_trends.py` | 运行趋势聚合 |
| `examples/sixdst_custom.py` | 自定义算法封装示例 |

示例命令：

```bash
python examples/run_sixdst.py --dataset ForestFire_n500 --mode s
python examples/run_sixdst.py --dataset ForestFire_n500 --mode m --server 6 --use-strategy-plan
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

## 运行时配置

- `servers.json`：远程服务器清单与端点
- `algorithms.json`：通用/自定义算法注册入口

常用环境变量：

- `GAPA_MNM_MAX_WORKERS`（默认 `4`）
- `GAPA_MNM_REFRESH_S`（默认 `2.0`）
- `GAPA_MNM_MIN_CHUNK_SIZE`（默认 `6`）
- `GAPA_MNM_COMM_WINDOW_ITERS`（默认 `3`）
- `GAPA_MNM_FP16_MIN_ROWS`（默认 `24`）
- `GAPA_RPC_COMPRESS_MIN_BYTES`（默认 `2048`）
- `GAPA_RPC_COMPRESS_MIN_SAVING`（默认 `0.05`）
- `GAPA_RPC_TORCH_LEGACY_SERIALIZATION`（默认 `1`）

## 自定义算法接入

用户自定义或通用算法建议统一通过 `algorithms.json` 注册。

推荐流程：

1. 在 `examples/` 或你自己的包中实现算法封装。
2. 在 `algorithms.json` 中登记 `entry`、`init_kwargs` 和 `capabilities`。
3. 通过 `Workflow` 或示例脚本发起运行。
4. 如果要走远程或 MNM，确保本地 `app.py` 与远端 `server_agent.py` 都能导入对应入口。
