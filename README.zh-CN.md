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

## 运行时契约（跨平台）

`/api/analysis/start` 与 `/api/analysis/status` 统一返回 `mode_decision`：

- `requested_mode`
- `selected_mode`
- `degraded`
- `reason`
- `target`
- `devices`

当请求模式不可用时，严格按 `MNM -> M -> SM -> S`（或其前缀）降级，并返回可追踪 `reason`。
任务终态统一为：`completed`、`error`、`cancelled`。

针对网络抖动或部分节点离线：
- `analysis/*` 与 `resource_lock*` 已统一接入重试、超时与结构化错误码。
- 可通过 `GET /api/transport/metrics` 导出诊断指标（失败率、重试次数、降级原因、平均恢复时长）。

恢复优先流程（兼容 schema 版本）：
- `POST /api/analysis/start` 支持 `schema_version`（默认 `v1`，扩展为 `v2`）、`checkpoint_ref`、`retry_last`。
- `start/status/stop` 统一返回 `schema_version`、`run_id`、`resume_metadata` 与标准化 `mode_decision`（`degraded/reason/code`）。
- legacy `/api/state` 保留旧 `status` 字段，同时新增规范化 `state` 与 `is_terminal`。

队列持久化与重启恢复：
- Busy 场景进入队列的任务会持久化（`task_id/owner/priority/payload/retry_count/created_at`）。
- 服务重启会恢复待执行队列，并跳过已知终态任务，避免重复回放。
- 队列响应统一包含 `position/status/error_code` 字段。

前端控制台模块化：
- `web/assets/app.js` 已改为依赖模块化助手：
  - `web/assets/api-client.js`（统一超时/重试）
  - `web/assets/ui-state.js`（共享状态存储）
  - `web/assets/ui-render.js`（模式/降级信息渲染）
- 控制台 Header 新增全局语言切换（`zh-CN` / `en`）。
- 语言选择会持久化到 `localStorage` 的 `gapa_lang`，刷新页面后保持上次选择。
- 语言切换仅影响 UI 文案，不改变 API 契约、脚本示例行为和运行时请求参数。

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
