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
| Qwen3-32B | **1.0187×** | 1.0207× | 1.0187× | 1.0185× |
| Llama-3.3-70B-Instruct | **1.0065×** | 1.0026× | 1.0066× | 1.0065× |

**两个模型的数据成色不同，必须分开讲**（详见 9.2）：

- **Llama 六格是干净的**：每格 n≥2、两臂配置一致、背靠背采集、autotune 覆盖对称。
  但 +0.65% 小于实测 1.93% 的噪声地板，诚实的结论是**无显著差异**，不是"加速 0.65%"。
- **Qwen 五格为单次测量**（c=8/32/64/128/256，只有 c=16 是 n=2），且这五格两臂的
  Triton 搜索空间与 autotune 覆盖不对称。**1.0187× 不具备独立发表条件**，
  引用时必须注明是单次测量。

第 8 节解释机制，9.2 列出全部已知缺陷。

---

## 1. 测试矩阵

| 维度 | 取值 | 点数 |
|---|---|---|
| 模型 | Qwen3-32B、Llama-3.3-70B-Instruct | 2 |
| dtype | bfloat16 | 1 |
| 工作负载 | showcase = ISL 256 / OSL 512 | 1 |
| 并发 | 8, 16, 32, 64, 128, 256 | 6 |
| 臂 | baseline = `ATEN,TRITON`；treatment = `ATEN,TRITON,FLYDSL` | 2 |
| 重复 | n = 2，ABBA 交错 | 2 |

配方要求 2 × 6 × 2 × 2 = **48 个点**。**已发布的这一批是 42 个点**：
Llama 六格全部 n≥2（c=16/c=64 为 n=3，其余 n=2），Qwen 只有 c=16 是 n=2、
其余五格 n=1。Qwen 的缺口是历史遗留，不是配方的一部分，详见 9.2。

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

**实测噪声：** 把全部 14 组有重复的测量合并，单次运行在 log 空间的组内标准差是
**1.93%**。n=2 时一格比值的标准误约等于这个数，所以**单格 2% 以内的差异都只有 1σ**。
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
| `E2E_MAX_MODEL_LEN` | 4096 |
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
| Qwen3-32B | +3.5% | −0.8% | +2.7% | +0.3% | +3.0% | +2.5% | 1.0187× |
| Llama-70B | +1.1% | +0.3% | +0.5% | +0.5% | +1.2% | +0.3% | 1.0065× |

TTFT：

| 模型 | c=8 | c=16 | c=32 | c=64 | c=128 | c=256 | 几何平均 |
|---|---|---|---|---|---|---|---|
| Qwen3-32B | −0.4% | −2.1% | −0.2% | +4.4% | +2.6% | +8.3% | 1.0207× |
| Llama-70B | −1.3% | +1.3% | +0.4% | +0.4% | +0.1% | +0.6% | 1.0026× |

### 8.2 路由取证：FlyDSL 确实被选中了

从 server 日志统计 autotune 决策的胜出者（胜出者是 `dtypes:` 行之后的**第一条
候选**，不是 `AUTOTUNE` 头之后的第一行——这里写错会得出"FlyDSL 从未胜出"的
错误结论）：

| 模型 | 臂 | 决策数 | FlyDSL 胜 | ATen 胜 | Triton 胜 | FlyDSL 胜出时中位领先 |
|---|---|---|---|---|---|---|
| Qwen3-32B | baseline | 53 | 0 | 43 | 10 | — |
| Qwen3-32B | treatment | 100 | **28 (28%)** | 69 | 3 | **+6.59%** |
| Llama-70B | baseline | 182 | 0 | 159 | 23 | — |
| Llama-70B | treatment | 182 | **76 (42%)** | 106 | 0 | **+3.29%** |

两个 baseline 臂的 FlyDSL 胜出均为 0，符合预期（候选列表里没有它）。

### 8.3 为什么 Llama 的端到端收益远小于 Qwen

两条独立证据：

**(a) kernel 层优势本身更小。** Llama 上 FlyDSL 胜出时中位领先 +3.29%，
Qwen 是 +6.59%，差 2 倍。

**(b) Llama 的 decode 已经贴着 HBM 带宽天花板。** 按每 decode step 必读的
权重 + KV 字节数除以实测流式带宽估算下限：

| 并发 | Llama 带宽利用率 | Qwen 带宽利用率 |
|---|---|---|
| 8 | **83%** | 67% |
| 32 | 77% | 61% |
| 256 | 47% | 39% |

Llama 在低并发只剩约 17% 余量，且余量里还包含 attention、采样和调度。
FlyDSL 赢下的是小 M 的 decode GEMM，这些算子本身就是带宽受限的——
kernel 层赢 3.57%，换不成端到端收益。Qwen 余量大得多，所以同样的机制能兑现。

**结论的正确表述是"Llama 上两臂无显著差异"，不是"FlyDSL 让 Llama 变快了"**：
Llama 六档 |Δ| 全部 ≤ 1.2%，而噪声地板是 1.93%——全部在噪声之内。

### 8.4 统计功效的边界

**噪声地板：把全部 14 组有重复的测量合并，单次运行在 log 空间的组内标准差是
1.93%。** n=2 时一格比值的标准误约等于这个数。

据此判定每一格：

| 模型 | 逐格 \|Δ\| 范围 | 是否超出 1.93% |
|---|---|---|
| Llama-70B | 0.3% – 1.2% | 六格**全部不超出** |
| Qwen3-32B | 0.3% – 3.5% | c=8 (+3.5%)、c=128 (+3.0%) 超出；其余不超出 |

**Llama 面板的 TPOT 差异在本样本量下完全不可分辨**，正确表述是"两臂无显著差异"。

Qwen 有两格超出噪声地板，但**那两格都是 n=1**——单次测量无法估计自身的不确定度，
超出"用别的格测出来的噪声地板"不等于统计显著。Qwen 的 `*_spread` 在五格里恒为 0，
那是"没测"而不是"很稳"。

重复测量如何改变结论，有三个实例值得记住：

| 格 | 单次/n=2 时的结论 | 补测后 | 摆动 |
|---|---|---|---|
| Llama c=8 | −3.4%（n=1） | +1.1%（n=2） | 4.5 pp |
| Llama c=16 | +3.8%（n=2） | +0.3%（n=3） | 3.5 pp |
| Llama c=32 | −1.2%（n=2） | +0.5%（重测 n=2） | 1.7 pp |

Llama c=16 那次尤其能说明问题：baseline 三次测得 27.93 / 30.14 / 28.03，
中位数把 30.14 这个孤立异常值排除后，"+3.8% 的加速"变成了"+0.3%，无差异"。
**若按 n=2 交付，它会以比 Qwen 最好一格还高的数字进图。**

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

### 9.2 已发布这一批数据的缺陷

**Llama 面板已全部修复，Qwen 面板仍有缺陷。** 这是交付时必须分开陈述的。

| 检查项 | Llama-70B | Qwen3-32B |
|---|---|---|
| 样本量 | 六格全部 n≥2（c=16/c=64 为 n=3） | **五格 n=1**，只有 c=16 是 n=2 |
| 两臂 Triton 搜索空间 | 六格全部 DEFAULT，对称 | **五格为 EXHAUSTIVE 或两臂不对称** |
| autotune 覆盖对称性 | 六格全部对称（26/26、39/39…） | **五格不对称**，c=128 的 baseline 日志里零次 autotune |
| 两臂采集间隔 | 0.01–0.21 小时，背靠背 | **4.6–27.2 小时**，跨批次 |
| KV 容量两臂一致 | 是 | 是 |

Qwen 的三项缺陷有共同成因：那五格的 baseline 采集于加入
`E2E_TRITON_DEFAULT_SPACE` 开关之前，treatment 则是之后补测的，
而 `run_e2e.sh` 的断点续跑逻辑（"all points present, skipping"）在补测时
跳过了已存在的 baseline，只重跑 treatment，从而绕开了分片设计本应提供的
同卡、同时段配对。

**Triton 空间不对称的偏置方向对 treatment 不利**（baseline 拿到 5160 个候选、
treatment 只有 36 个），所以 Qwen 的收益是低估而非高估。但"偏保守"不等于
"可发表"——该对照本身不成立。

**交付建议**：Llama 面板可独立交付；Qwen 的 1.0187× 引用时必须注明
"五格为单次测量，两臂 autotune 配置不对称"。

### 9.3 未定位的问题

`hipErrorInvalidValue` 在长 sweep 的第 13–14 轮 KV 分配期间出现过 3 次，
**根因未找到**。supervisor 会自动补跑受影响的点位。

### 9.4 要让 Qwen 面板也可发表，需要做什么

一次性重跑 Qwen 全部六格，不要增量拼接：

```bash
E2E_MODELS="Qwen/Qwen3-32B" E2E_REPEATS=2 bash scripts/run_all.sh
```

12 个分片，8 卡并行约 1 小时。配方默认已是 `E2E_TRITON_DEFAULT_SPACE=1`，
单批采集，两臂自然背靠背，一次解决上表中 Qwen 的全部四项。
