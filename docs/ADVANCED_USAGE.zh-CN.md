# 进阶使用

这个文档承接 README 中不适合放在首页新手主路径里的内容，包括服务启动、远程运行时和进阶集成说明。

文档导航：

- [README.zh-CN.md](README.zh-CN.md)

## 公开入口

当前维护的公开示例入口只有两层：

- `examples/api/`
- `examples/algorithms/`

历史脚本式参考保留在 `old_examples/`，仅作兼容与留档用途。

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

- `.env` / `.env.example`：服务地址、远程端点、资源筛选条件
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

1. 在你自己的包中实现算法封装，或从 `examples/api/algorithm.py` 起步。
2. 在 `algorithms.json` 中登记 `entry`、`init_kwargs` 和 `capabilities`。
3. 通过 `Workflow` 发起运行。
4. 如果要走远程或 MNM，确保本地 `app.py` 与远端 `server_agent.py` 都能导入对应入口。
