# 端到端 benchmark 方案手册（定稿）

本文件记录**已经跑通并定稿**的端到端 benchmark 配方：每个参数取什么值、为什么
取这个值、跑出来是什么结果、以及结果能支持到什么程度的结论。

仓库的整体设计见 [README.md](README.md)；给自动化工具用的复现步骤见
[.claude/REPRODUCE.md](.claude/REPRODUCE.md)。

---

## 0. 一句话结论

在 ISL=256 / OSL=512、并发 8–256 的服务端工作负载上，把 FlyDSL 加入 Inductor 的
GEMM autotune 候选列表：

| 模型 | TPOT | TTFT | 端到端延迟 | 输出吞吐 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | **1.0692×** | 1.0150× | 1.0623× | 1.0620× |
| Qwen3-32B | **1.0173×** | 1.0106× | 1.0169× | 1.0168× |
| Llama-3.3-70B-Instruct | **1.0065×** | 1.0026× | 1.0066× | 1.0065× |

**十八格全部满足采集条件**：每格 n≥2、两臂同配置（均 Triton DEFAULT）、
背靠背跑在同一张卡上、autotune 覆盖对称、KV 容量两臂相同。

噪声地板是 **1.79%**（36 组重复合并）。据此逐格判定：**Llama-3.1-8B 有 5 格
超出（4 正 1 负），Qwen3-32B 有 1 格（c=64），Llama-3.3-70B 一格都没有**——
后者的正确表述是"两臂无显著差异"，不是"加速 0.65%"。

三个模型跨 10 倍尺寸给出一条干净的规律：**decode 的带宽余量决定 kernel 改进
能兑现多少**，见 8.3。第 8 节解释机制，9.2 记录定稿前修掉的采集问题。

---

## 1. 测试矩阵

| 维度 | 取值 | 点数 |
|---|---|---|
| 模型 | Llama-3.1-8B-Instruct、Qwen3-32B、Llama-3.3-70B-Instruct | 3 |
| dtype | bfloat16 | 1 |
| 工作负载 | showcase = ISL 256 / OSL 512 | 1 |
| 并发 | 8, 16, 32, 64, 128, 256 | 6 |
| 臂 | baseline = `ATEN,TRITON`；treatment = `ATEN,TRITON,FLYDSL` | 2 |
| 重复 | n = 2，ABBA 交错 | 2 |

配方的下限是 2 × 6 × 2 × 2 = 48 个点。**已发布的这一批是 56 个点**：
基础 n=2，另有四格补到更高重复——Llama 的 c=32/c=64 为 n=3，
Qwen 的 c=64 baseline 为 n=6（该格效应最大，加测以缩小不确定度）。

---

## 2. 参数是怎么定下来的

### 2.1 并发档位决定一切

decode 阶段 GEMM 的 M 维**就等于并发数**。所以并发档位不是"顺便扫一下"，
它直接决定被测 kernel 的形状。8–256 覆盖了 FlyDSL 相对基线优势最大的区间。

### 2.2 `max_num_batched_tokens` = 2048（最容易被误解的一项）

prefill 的 GEMM 宽度是 `min(ISL × 并发, max_num_batched_tokens)`，**不是 ISL**。

这意味着：想让 TTFT 反映真实的 prefill GEMM，要调的是这个上限和并发，
而不是一味加大 ISL。本方案把它钉死在 2048——vLLM 的 offline 与 server 两条
代码路径默认值不同，不钉死则"换个入口跑出来的数不一样"。

### 2.3 ISL = 256 / OSL = 512

- ISL=256 配合并发 8 起步，最窄的 prefill 也有 2048 token，够得着 GEMM；
  ISL 再小（如 32），TTFT 测的就是调度开销而非 GEMM，`plot.py` 会直接拒绝出图。
- OSL=512 让每个 prefill token 对应 2 个 decode step，稳态由 decode 主导，
  TPOT 成为主口径。

### 2.4 搜索空间：FlyDSL EXHAUSTIVE + Triton DEFAULT

`E2E_AUTOTUNE_SEARCH_SPACE=EXHAUSTIVE` 配合 `E2E_TRITON_DEFAULT_SPACE=1`。

探索中实测：gfx950 上 bf16 GEMM 的候选数，关闭调优为 1、DEFAULT 为 34、
EXHAUSTIVE 为 1998（48384 组合中 4.13% 通过约束）。Triton 的 EXHAUSTIVE 在
M≥128 时达 4080 个候选，按实测 0.20 s/候选计要多花约 816 s 的纯预编译，
而在本方案涉及的形状上，它选出的 kernel 与 DEFAULT 没有可测差异。

更关键的是：`TORCHINDUCTOR_PRECOMPILATION_TIMEOUT_SECONDS` 默认 300 s，
超时的候选会被静默丢弃，**幸存候选集因此取决于机器负载而非配置**，不可复现。
本方案把它提到 3600 s，同时用 Triton DEFAULT 把候选数压到不会触发超时的量级。

### 2.5 显存：`E2E_GPU_MEM_UTIL=0.85` + KV 容量钉死

KV 容量由 `calibrate_kv_all.sh` 逐模型标定后写入 `state/kv_pin.d/<MODEL>.env`，
两臂强制相同（Llama 实测 94.16 GB / 287328 tokens）。不钉死的话，拿到更多 KV
块的一臂会因为与 GEMM 无关的原因赢在吞吐上。

util 取 0.85：更高会挤压编译工作区，在 70B 上触发过 KV 分配失败。

### 2.6 重复次数 n = 2 —— 本方案最重要的一条

**实测噪声：** 把全部 36 组有重复的测量合并，单次运行在 log 空间的组内标准差是
**1.79%**。n=2 时一格比值的标准误约等于这个数，所以**单格 1.7% 以内的差异都只有 1σ**。
以极差计更直观：单臂重复极差最大到 7.7%（Llama c=16 的 baseline），TTFT 更差。

这意味着 n=1 分辨不出 3% 以内的任何差异，连正负号都会翻转。定稿前有一轮
n=1 的数据给出"Llama TPOT 变慢 1.2%"，同配置重测后是 +0.5%，其中 c=8 从
−3.4% 翻到 +1.1%，摆动 4.5 个百分点——**整个结论是测量假象**。

注意 `config/default.env` 里 `E2E_REPEATS` 的默认值必须是 2。它曾经是 1，
导致方案文档写 n=2 而实际只跑了 r1。跑完务必核对：

```bash
ls runs/<run_id>/raw/e2e/ | grep -c _r2
```

---

## 3. 固定配置

以下值在 `config/default.env` 中即为默认，不需要额外设置：

| 参数 | 值 |
|---|---|
| `E2E_TP` | 1 |
| `E2E_MAX_NUM_BATCHED_TOKENS` | 2048 |
| `E2E_MAX_MODEL_LEN` | 1024 |
| `E2E_MAX_NUM_SEQS` | 256 |
| `E2E_GPU_MEM_UTIL` | 0.85 |
| `E2E_REPEATS` | 2 |
| `E2E_AUTOTUNE_SEARCH_SPACE` | EXHAUSTIVE |
| `E2E_TRITON_DEFAULT_SPACE` | 1 |
| `E2E_FLYDSL_AUTOTUNING` | 1 |
| `E2E_PRECOMPILE_TIMEOUT_S` | 3600 |
| `E2E_SERVER_TIMEOUT_S` | 43200 |
| `E2E_SERVER_STALL_S` | 5400 |
| 客户端 | `vllm bench serve`，prefix caching 关闭，`ignore_eos`，seed 0 |

TP=1 是刻意的：切分会把每个 Linear 的 N 或 K 除以 TP 数，被测 GEMM 宽度变成
真实值的几分之一，同时给 decode 关键路径加上两臂等价的集合通信，只会稀释信号。
TP=1 不代表只用一张卡——`run_sharded.sh` 以单卡独立分片的方式占满所有可用 GPU。

---

## 4. 缓存策略

每个点位启动前，`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` /
`VLLM_CACHE_ROOT` 三个目录**整体重建**，保证 autotune 真实发生。

删除方式是"改名 + 后台删除"，不是就地 `rm -rf`：在网络文件系统上，刚被杀死的
进程会留下仍被打开的文件句柄（`.nfsXXXX`），就地删除会失败，而在 `set -e` 下
这个失败会静默带走整个 sweep。

---

## 5. 时间预估

单点实测（Llama-70B、Triton DEFAULT）：

| 臂 | 单点耗时 | 构成 |
|---|---|---|
| baseline | ~8 分钟 | 权重加载 + 编译 + 92 s 压测 |
| treatment | ~20 分钟 | 额外的 FlyDSL 候选编译 |

分片粒度是 (模型, 并发, 重复)，两臂在同一张卡上背靠背跑，所以一个分片 ~28 分钟。
Llama 单模型 4 并发 × 2 重复 = 8 分片，8 卡并行约 **35–60 分钟**（含 NFS 权重
加载争用）。全矩阵 2 模型 × 6 并发 × 2 重复 = 24 分片，8 卡约 **1.5–2 小时**。

注意 FlyDSL 的 `FlyDSLTemplateCaller` 没有 `precompile` 方法，因而不会进入
Inductor 的预编译线程池（`select_algorithm.py` 里以 `hasattr(c, "precompile")`
为门），它在 benchmark 阶段惰性编译，每轮额外约 62 s。这是 treatment 臂更慢的
原因，与 kernel 性能无关。

---

## 6. 执行步骤

```bash
export E2E_VENV=/path/to/python/env
export E2E_VLLM_SRC=/path/to/vllm

bash scripts/run_all.sh                 # 00–10 全流程
bash scripts/supervisor.sh &            # 守护：只补跑没有活分片覆盖的点位
bash scripts/progress.sh                # 查看进度
```

只跑后处理：

```bash
E2E_ONLY="07 08 09 10" bash scripts/run_all.sh
```

只补测某几个点位（例如只重测 Llama 的 4 个并发档）：

```bash
E2E_MODELS="meta-llama/Llama-3.3-70B-Instruct" \
E2E_CONCURRENCY_SWEEP="8 32 128 256" \
RUN_DIR="$PWD/runs/<new_run_id>" \
  bash scripts/bench/run_sharded.sh
```

---

## 7. 产物

```text
runs/<run_id>/
  raw/e2e/*.json                每点一份，含完整 workload 描述与 provenance
  normalized/e2e_bench.csv      逐重复
  normalized/e2e_agg.csv        逐 cell，每个指标带 *_mean 与 *_spread
  figures/*.png                 每张带 JSON sidecar
  logs/server_<arm>_<model>_<dtype>_<profile>_r<rep>_c<conc>.log

result/
  <Model>_<dtype>_exhaustive/   每模型 8 张图 + summary.csv + raw_points.csv
  bf16_result_png/              两模型合并的对比图，每个指标一张
```

---

## 8. 结果与解读

### 8.1 逐档数据

TPOT（正数 = treatment 更快）：

| 模型 | c=8 | c=16 | c=32 | c=64 | c=128 | c=256 | 几何平均 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | **+11.9%** | **+15.7%** | **+9.4%** | **+8.9%** | −0.3% | **−2.9%** | **1.0692×** |
| Qwen3-32B | +0.2% | −0.8% | +1.4% | **+5.7%** | +1.7% | +2.2% | 1.0173× |
| Llama-70B | +1.1% | +0.3% | +0.5% | +0.5% | +1.2% | +0.3% | 1.0065× |

TTFT：

| 模型 | c=8 | c=16 | c=32 | c=64 | c=128 | c=256 | 几何平均 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | +4.2% | −2.6% | +1.2% | −3.0% | −3.9% | +14.3% | 1.0150× |
| Qwen3-32B | −1.0% | −2.1% | +1.6% | +2.9% | +3.7% | +1.4% | 1.0106× |
| Llama-70B | −1.3% | +1.3% | +0.4% | +0.4% | +0.1% | +0.6% | 1.0026× |

### 8.2 路由取证：FlyDSL 确实被选中了

从 server 日志统计 autotune 决策的胜出者（胜出者是 `dtypes:` 行之后的**第一条
候选**，不是 `AUTOTUNE` 头之后的第一行——这里写错会得出"FlyDSL 从未胜出"的
错误结论）：

| 模型 | 臂 | 决策数 | FlyDSL 胜 | ATen 胜 | Triton 胜 | FlyDSL 胜出时中位领先 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | treatment | 156 | **95 (61%)** | 61 | 0 | decode +2.48% / prefill +7.99% |
| Qwen3-32B | treatment | 168 | **51 (30%)** | 115 | 2 | decode +0.54% / prefill +6.98% |
| Llama-70B | treatment | 182 | **76 (42%)** | 106 | 0 | decode +2.90% / prefill +3.32% |

三个 baseline 臂的 FlyDSL 胜出均为 0，符合预期（候选列表里没有它）。
**领先幅度必须按形状分开看**，合并后的中位数会被 prefill 样本主导——见 8.3。



### 8.3 决定收益大小的是 decode 的带宽余量

三个模型跨 10 倍尺寸给出一条干净的规律：

| 模型 | decode 带宽利用率 | FlyDSL 赢下的 decode 决策 | TPOT 几何平均 |
|---|---|---|---|
| Llama-3.1-8B | **23–39%** | **47 / 48（98%）** | **1.0692×** |
| Qwen3-32B | 39–67% | 25 / 48（52%） | 1.0173× |
| Llama-3.3-70B | 47–83% | 33 / 56（59%） | 1.0065× |

**模型越小、带宽余量越大，kernel 改进兑现得越多。**

Llama-3.3-70B 的 decode 已经贴着 HBM 天花板：实测 decode 线性层 21.74 ms，
理论带宽下限 21.73 ms（**100%**）。在这种状态下 kernel 再快也搬不动显存。
一个量化的佐证：该模型上 FlyDSL 的 decode GEMM 在 autotune 台面上领先
1.5%–11.2%，**但只有约 10–20% 兑现到端到端 TPOT**；c=64 那格台面 +11.16%、
端到端只剩 +0.50%，兑现率 4%。Llama-3.1-8B 没有这个损耗。

**注意不要混淆 prefill 与 decode 的统计量。** 本项目早先版本引用过一个把两者
合并后的"中位领先"，用它论证 decode——那是错的：可算 margin 的样本里绝大多数
落在 prefill 形状（M=2048）上。分开统计才有意义：

| 模型 | decode 胜/总 | decode 中位领先 | prefill 胜/总 | prefill 中位领先 |
|---|---|---|---|---|
| Llama-3.1-8B | 47/48 | +2.48% (n=11) | 48/108 | +7.99% (n=34) |
| Qwen3-32B | 25/48 | +0.54% (n=3) | 26/120 | +6.98% (n=26) |
| Llama-3.3-70B | 33/56 | +2.90% (n=5) | 43/126 | +3.32% (n=43) |

decode 形状的 margin 样本很少，因为多数 decode 块的 top-10 候选全是 FlyDSL，
没有可比的非-FlyDSL 亚军——这本身说明 FlyDSL 在 decode 候选池里占优。

### 8.4 统计功效的边界

**噪声地板：把全部 36 组有重复的测量合并，单次运行在 log 空间的组内标准差是
1.79%。** n=2 时一格比值的标准误约等于这个数。

据此判定每一格：

| 模型 | 逐格 \|Δ\| 范围 | 超出 1.79% 的格数 |
|---|---|---|
| **Llama-3.1-8B** | 0.3% – 15.7% | **5 / 6**（c=8/16/32/64 为正，c=256 为负） |
| Qwen3-32B | 0.2% – 5.7% | 1 / 6（c=64） |
| Llama-3.3-70B | 0.3% – 1.2% | **0 / 6** |

**Llama-3.3-70B 的正确表述是"两臂无显著差异"。** Llama-3.1-8B 的结论最硬：
五格超出噪声，且六格的两次重复**符号全部一致**。

重复测量如何改变结论，四个实例值得记住：

| 格 | 少量重复时 | 补测后 | 摆动 |
|---|---|---|---|
| Llama-70B c=8 | −3.4%（n=1） | +1.1%（n=2） | 4.5 pp |
| Llama-70B c=16 | +3.8%（n=2） | +0.3%（n=3） | 3.5 pp |
| Qwen c=8 | +3.5%（n=1） | +0.2%（n=2） | 3.3 pp |
| Qwen c=64 | +0.3%（n=1） | +5.7%（baseline n=6） | 5.4 pp |

最后一个是反方向的例子：n=1 时测得 +0.3%，看着毫无收益；baseline 补到 6 次后
稳定在 20.62 ms（sd 1.9%），原来那个孤立的 19.58 距均值 −2.32σ，是异常值。
**n=1 既会造出不存在的收益，也会抹掉真实的收益。**

### 8.5 其他已知局限

1. **结论是有条件的。** 它成立的前提是走 Inductor 的 autotune 路径
   （`VLLM_FORCE_ATEN_LINEAR=1`）。ROCm 上 vLLM 默认不走 `aten.mm`，而是
   分派到自带的 kernel 选择树，Inductor 看不进自定义算子。这个开关**两臂都开**，
   它不是 A/B 的一部分，而是让 A/B 成为可能的前提。
2. **只覆盖 `mm`，不覆盖 `addmm`。** 本方案的两个模型都没有 bias
   （`attention_bias: False`、`mlp_bias: False`），因此不产生 `addmm`。
   换成带 bias 的模型前要先确认后端是否 hook 了 `addmm`。
3. **`lm_head` 不参与 autotune**，占 decode 权重流量的 1.5–2.4%，两臂相同。
4. **单机单次 sweep**，没有跨机器复现。

---

## 9. 定稿过程中的返工记录

留作后来者的避坑清单。

### 9.1 已修复的方法学问题

| 问题 | 后果 | 修复 |
|---|---|---|
| `E2E_REPEATS` 默认 1 | 方案写 n=2，实际全是 r1，结论是噪声 | 默认改为 2，跑完核对 `_r2` 计数 |
| `run_all.sh` 的 cleanup 按项目路径匹配进程 | 退出时杀掉了手动启动的分片和 supervisor | 改为只杀本实例的子孙进程 |
| `daemon/stop.sh` 的 pattern sweep | 同上 | 默认关闭，需 `E2E_STOP_SWEEP=1` 显式开启 |
| 缓存目录就地 `rm -rf` | NFS 句柄导致失败，`set -e` 下静默带走整个 sweep | 改名 + 后台删除 |
| stall 阈值 900 s | 短于 EXHAUSTIVE 预编译的 1394 s 静默期，健康服务被杀 | 提到 5400 s |
| `E2E_SERVER_TIMEOUT_S` 3600 s | 短于实际需要，每轮服务在第二轮被杀 | 提到 43200 s |
| 解析 autotune 胜出者时取 `AUTOTUNE` 后第 1 行 | 取到 `strides:` 行，得出"FlyDSL 从未胜出"的错误结论 | 取 `dtypes:` 之后的第一条候选 |

### 9.2 定稿前修掉的采集问题

早期批次存在四类问题，**当前这批已全部修复**，记录下来是为了让后来者认得出它们。

| 问题 | 当时的表现 | 修法 |
|---|---|---|
| 样本量不足 | 多数格 n=1，逐格数值不可信、符号会翻转 | 全部补到 n≥2，效应最大的格补到 n=6 |
| 两臂 Triton 搜索空间不对称 | baseline 5160 候选 vs treatment 36，相差 143 倍 | 两臂统一 Triton DEFAULT |
| autotune 覆盖不对称 | 某些 baseline 日志里 autotune 块为 0（缓存命中或日志被覆盖） | 单批重跑，两臂覆盖对称 |
| 两臂非背靠背采集 | 增量补跑使两臂相隔 4.6–27.2 小时 | 单批采集，同卡背靠背 |

第三、四项的共同成因是 `run_e2e.sh` 的断点续跑逻辑（"all points present,
skipping"）：事后手动补测时它跳过已存在的 baseline、只重跑 treatment，
绕开了分片设计本应提供的同卡、同时段配对。**增量拼接是这套 harness 最容易
踩的坑**——要改配置就整批重跑，不要只补一半。

**当前批次的验收状态**：十二格全部 n≥2、两臂均 Triton DEFAULT、背靠背同卡、
autotune 覆盖对称、KV 容量两臂相同。

### 9.3 Triton 搜索空间对 baseline 的影响有多大

配方选 Triton DEFAULT 的前提是"两个搜索空间选出的 kernel 没有可测差异"。
这一点被直接验证过：取 Qwen c=256 baseline 臂在 EXHAUSTIVE（5160 候选）与
DEFAULT（36 候选）下的日志，逐 shape 比较胜出 kernel——

- **8 个 shape 里 7 个选出同一个 kernel**，其中 6 个的胜出者是 ATen 的 `mm`，
  Triton 根本没赢，搜索空间自然无关紧要；
- 只有一个 shape 真正受影响（`triton_mm_1900` vs `triton_mm_130`，差 8.5%）；
- **合计下来 DEFAULT 只慢 0.8%。**

所以用 DEFAULT 换掉 EXHAUSTIVE，代价是 baseline 慢约 0.8%，而收益是每个模型
省下约 816 秒的纯预编译、且候选集不再受预编译超时影响而依赖机器负载。

### 9.4 未定位的问题

`hipErrorInvalidValue` 在长 sweep 的第 13–14 轮 KV 分配期间出现过 3 次，
**根因未找到**。supervisor 会自动补跑受影响的点位。

### 9.5 换配置时的注意事项

任何改变两臂可比性的配置变更（搜索空间、KV 容量、MNBT、dtype），
都要**整批重跑**，不能只补一半：

```bash
E2E_MODELS="<model>" E2E_REPEATS=2 bash scripts/run_all.sh
```

单模型 12 个分片，8 卡并行约 1 小时。单批采集才能保证两臂背靠背、
autotune 覆盖对称。9.2 表里的三类问题都是增量拼接造成的。
