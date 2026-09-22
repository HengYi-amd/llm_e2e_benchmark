# MXFP8 / MXFP4 端到端 benchmark 方案

本文件是 **mxfp8 与 mxfp4 scaled GEMM** 端到端 benchmark 的执行方案。
bf16 那一轮的定稿配方见 [RUN_Recipe_bf16.md](RUN_Recipe_bf16.md)。

**与 bf16 轮次的关系**：模型、并发、ISL/OSL、重复次数、出图方式**全部保持一致**，
唯一变化是被测算子从 `mm` 变成 `_scaled_mm_v2`。三种精度的结果可直接横向比较。

本方案中标注 **[已验证]** 的结论都在本机实测过，命令与输出可复现；
标注 **[待验证]** 的是尚未上机确认的部分。

---

## 0. 结论先行

三个阻碍全部有了确定的解法，其中两个比最初判断的要**简单**：

| # | 问题 | 结论 |
|---|---|---|
| 1 | PyTorch 缺 mxfp 路径 | 必须整分支安装，**文件覆盖行不通**（已实测失败并回滚） |
| 2 | vLLM MX 线性层不走 `_scaled_mm_v2` | 需要新增一个 linear kernel 类，vLLM 的注册表结构使这件事很干净 |
| 3 | scale layout 冲突 | **实际不存在**——dense 线性层的激活量化默认就不 swizzle |

隔离机制已经落地：`E2E_PRECISION` 一个变量同时切换 **venv、torch 源码树、KV pin
命名空间、交付目录**，bf16 已发布的数据与环境不可能被 mxfp 影响。

---

## 1. 被测算子的确切契约 [已验证]

来自 PR #196719 的 `_get_rocm_mxfp_v2_format()`。**以下条件必须全部满足**，
缺一条 FlyDSL 就不会产生候选，且是静默的：

```python
torch.version.hip is not None                        # ROCm ✓
len(scale_a) == 1 and len(scale_b) == 1
recipe_a == recipe_b == [ScalingType.BlockWise1x32.value]
swizzle_a == swizzle_b == [SwizzleType.NO_SWIZZLE.value]
scale_a[0].dtype == scale_b[0].dtype == torch.float8_e8m0fnu
out_dtype in (torch.bfloat16, torch.float16)
_flydsl_mxfp_bias_supported(bias, mat_b, out_dtype)
not contraction_dim
use_fast_accum is False
mat_a.dtype == mat_b.dtype ∈ {float8_e4m3fn → "mxfp8",
                              float4_e2m1fn_x2 → "mxfp4"}
```

bias 的额外约束（`_flydsl_mxfp_bias_supported`）：`None`，或者
**1 维、与 mat_b 同设备、dtype 等于 out_dtype、长度等于 N**。

调用形式：

```python
torch._scaled_mm_v2(
    a, b.t(),                                   # A [M,K] row-major, B [K,N] col-major
    [scale_a], recipe, swizzle,                 # recipe=[BlockWise1x32], swizzle=[NO_SWIZZLE]
    [scale_b], recipe, swizzle,
    bias, [],                                   # contraction_dim 必须为空
    torch.bfloat16, False,                      # out_dtype, use_fast_accum
)
```

### 两臂在这条路径上的真实行为 [已验证]

`get_flydsl_mxfp_template_kwargs()` 第一行就是 `if not use_flydsl_gemm_template(layout): return []`
——标准的后端列表门禁。因此：

| 臂 | 后端列表 | MX GEMM 实际走什么 |
|---|---|---|
| baseline | `ATEN,TRITON,CK` | FlyDSL 候选为空 → 落到 `fallback()` → **纯 ATen `_scaled_mm_v2`，不 autotune** |
| treatment | `ATEN,TRITON,CK,FLYDSL` | FlyDSL 候选 + **代码主动插入的一个 ATen 候选** → 在两者间 autotune |

这依然是干净的超集对照：treatment 只会在 FlyDSL 实测更快时才改变结果。

**Triton 与 CK 在 MX 路径上不是候选**：stock 代码里写着
`# We don't have triton lowerings for the MX variants yet`，而 PR 的 MX 分支在
产生 FlyDSL 候选后直接返回，不再添加 CK。所以后端列表里写 `TRITON,CK` 只是
保持与 bf16 轮次的形式一致，**真实对照是 FlyDSL vs ATen**。这一点必须写进图注，
不能让读者以为 baseline 是调优过的 Triton/CK。

### 为什么 stock PyTorch 完全测不了 [已验证]

`tuned_scaled_mm_v2` 里有一道总门禁：

```python
disallowed = {ScalingType.BlockWise1x16, ScalingType.BlockWise1x32}
if (any(s != 0 for s in swizzle_a) or any(s != 0 for s in swizzle_b)
        or not supported_recipe):
    return fallback()          # 直接跳出，aten/triton/CK/flydsl 一个候选都不生成
```

**MX 的 `BlockWise1x32` 恰好在禁止列表里。** PR 的做法是在这道门禁**之前**插入
`_get_rocm_mxfp_v2_format()` 分支。所以在 stock PyTorch 上，mxfp8/mxfp4 连 autotune
都进不去——今天直接跑，两臂会是同一条 ATen fallback。

### 三个模型的形状全部合规 [已验证]

12 个线性层逐个核对 N%16、K%128、(K/2)%64，**全部通过**：

| 模型 | qkv_proj (N,K) | o_proj | gate_up | down_proj |
|---|---|---|---|---|
| Llama-3.1-8B | 6144, 4096 | 4096, 4096 | 28672, 4096 | 4096, 14336 |
| Qwen3-32B | 10240, 5120 | 5120, 8192 | 51200, 5120 | 5120, 25600 |
| Llama-3.3-70B | 10240, 8192 | 8192, 8192 | 57344, 8192 | 8192, 28672 |

prefill 的 M 恒为 `max_num_batched_tokens` = 2048（128 的倍数 ✓）。

**M 的约束存在两种说法**（算子层"M 为 16 的倍数" vs FlyDSL dispatch"M 无约束"）。
decode 的 M 等于并发数，8/16/32/64/128/256 **全部是 16 的倍数**，所以两种说法下
都安全——这个歧义在本方案里不构成风险，但换并发档位时要重新检查。

---

## 2. 阻碍 1：装 PyTorch —— 必须整分支，不能打补丁

### 已实测：文件覆盖行不通

PR 只改 6 个文件，且全是 `torch/_inductor/` 下的纯 Python，看起来可以直接覆盖。
**实测失败**：新 `mm.py` 依赖 313e0fb 不存在的 `_inductor` 内部 API
（`use_triton_tdm_template` 等），import 即报错。已回滚，bf16 环境经逐字节比对确认完好。

不要再尝试这条路。

### 正确做法：独立 worktree + 独立 venv

**关键原则：绝不动 bf16 的环境。** bf16 的数据是在 `pytorch_e2e` @ 313e0fb +
`.venv-e2e` 上采集的，而且那棵源码树里还有我们 bf16 轮次的未提交补丁
（`heuristics/template/triton.py` 的 Triton DEFAULT 开关等）。

```bash
# 1. 独立 worktree，不碰 pytorch_e2e
cd "$(dirname "$E2E_ROOT")/pytorch_e2e"
git fetch origin pull/196719/head:flydsl-mxfp          # [已完成]
git worktree add ../pytorch_mxfp flydsl-mxfp

# 2. 独立 venv 构建
E2E_PRECISION=mxfp8 bash scripts/build/build_torch.sh
```

`env.sh` 已改好：`E2E_PRECISION != bf16` 时自动指向 `pytorch_mxfp` 与 `.venv-mxfp`。[已验证]

### 安装后的验收 —— 不通过就不要往下走

```bash
E2E_PRECISION=mxfp8 bash -c '. env.sh; . "$E2E_VENV/bin/activate"; python - <<PY
import torch, os, inspect, re
base = os.path.dirname(torch.__file__)
k = os.path.join(base, "_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py")
s = open(k).read()
hits = {t: s.count(t) for t in ("mxfp", "e8m0", "e2m1", "f8f6f4")}
print("kernel 里的 mxfp 痕迹:", hits)
src = inspect.getsource(__import__("torch._inductor.kernel.mm", fromlist=["mm"]))
print("_get_rocm_mxfp_v2_format 存在:", "_get_rocm_mxfp_v2_format" in src)
print("get_flydsl_mxfp_template_kwargs 存在:", "get_flydsl_mxfp_template_kwargs" in src)
PY'
```

**验收标准**：三项全为真。**只要有一项不满足，后面全部无意义。**

### 微基准冒烟（上大模型之前必须过）

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON,CK,FLYDSL \
TORCHINDUCTOR_FLYDSL_AUTOTUNING=1 TORCH_LOGS=+inductor python - <<'PY'
import torch
from torch.nn.functional import ScalingType, SwizzleType
import torch._inductor.config as C
C.max_autotune = True; C.max_autotune_gemm = True
d, M, N, K = "cuda", 256, 4096, 4096
a = torch.randn(M, K, device=d).to(torch.float8_e4m3fn)
b = torch.randn(N, K, device=d).to(torch.float8_e4m3fn)
sa = torch.ones(M, K//32, device=d, dtype=torch.uint8).view(torch.float8_e8m0fnu)
sb = torch.ones(N, K//32, device=d, dtype=torch.uint8).view(torch.float8_e8m0fnu)
R = [ScalingType.BlockWise1x32.value]; S = [SwizzleType.NO_SWIZZLE.value]

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def f(a, b, sa, sb):
    return torch._scaled_mm_v2(a, b.t(), [sa], R, S, [sb], R, S,
                               None, [], torch.bfloat16, False)
print(f(a, b, sa, sb).shape)
PY
```

**期望看到** `AUTOTUNE _scaled_mm_v2(...)` 块里出现 `flydsl_*` 候选。
mxfp4 用 `torch.float4_e2m1fn_x2`、A 形状 `[M, K//2]` 再跑一遍。

---

## 3. 阻碍 2：把 vLLM 的 MX 线性层接到 `_scaled_mm_v2`

### 现状 [已验证]

vLLM 有一个清晰的 kernel 注册表 `vllm/model_executor/kernels/linear/`：

| 平台 | MXFP4 候选 | MXFP8 候选 |
|---|---|---|
| ROCM | `AiterMxfp4LinearKernel` → `EmulationMxfp4LinearKernel` | `RocmDotScaledMxfp8LinearKernel` → `EmulationMxfp8LinearKernel` |

两个原生实现都**不走** `_scaled_mm_v2`：

- `RocmDotScaledMxfp8LinearKernel`（`mxfp8/rocm_native.py`）：手写 Triton kernel，
  用 `tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")`
- `AiterMxfp4LinearKernel`（`mxfp4/aiter.py`）：`aiter.ops.triton.gemm_afp4wfp4`

两者对 Inductor 都不可见——和 bf16 轮次需要 `VLLM_FORCE_ATEN_LINEAR` 是同一类问题。

### 解法：新增一个 kernel 类，而不是改现有代码

vLLM 的注册表结构让这件事很干净，**不需要侵入式修改**：

1. 新建 `vllm/model_executor/kernels/linear/mxfp8/scaled_mm_v2.py`，
   实现 `ScaledMMv2Mxfp8LinearKernel(Mxfp8LinearKernel)`：

```python
class ScaledMMv2Mxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 linear through aten._scaled_mm_v2 so Inductor can autotune it.

    The native ROCm kernel calls a hand-written Triton kernel that Inductor
    cannot see through, which makes a GEMM-backend A/B impossible. This routes
    the same computation through the operator Inductor lowers, at the cost of
    leaving vLLM's own kernel out of the picture — so BOTH arms must use it.
    """

    @classmethod
    def is_supported(cls):
        # gfx950 only, and only when explicitly asked for: this exists to make
        # the backend measurable, not because it is the fastest path.
        return (on_gfx950() and envs.VLLM_FORCE_ATEN_SCALED_MM, None)

    def apply_weights(self, layer, x, bias=None):
        from torch.nn.functional import ScalingType, SwizzleType
        R = [ScalingType.BlockWise1x32.value]
        S = [SwizzleType.NO_SWIZZLE.value]
        x2d = x.reshape(-1, x.shape[-1])
        # is_sf_swizzled_layout defaults to False -> scales are already the
        # contiguous [M, K/32] layout the kernel contract asks for.
        x_q, x_scale = mxfp8_e4m3_quantize(x2d)
        out = torch._scaled_mm_v2(
            x_q,
            layer.weight.t(),                       # [N,K] -> [K,N] column-major
            [x_scale.view(torch.float8_e8m0fnu)],   # stored as uint8
            R, S,
            [layer.weight_scale.view(torch.float8_e8m0fnu)],
            R, S,
            None, [],                               # bias applied below, see note
            x.dtype, False,
        )
        out = out.reshape(*x.shape[:-1], layer.weight.shape[0])
        return out if bias is None else out + bias
```

2. 在 `_POSSIBLE_MXFP8_KERNELS[PlatformEnum.ROCM]` 里**插到第一位**。
   它的 `is_supported()` 由 `VLLM_FORCE_ATEN_SCALED_MM` 把关，
   不设这个变量时行为与今天完全一致。

3. mxfp4 同理，新建 `mxfp4/scaled_mm_v2.py`，A 用 `float4_e2m1fn_x2`、
   `[M, K//2]`，权重 `[N, K//2]` 传 `.t()`。

**和 bf16 轮次一样，这个开关两臂都开**——它不是 A/B 的一部分，而是让 A/B 成立的
前提。结论因此是**有条件的**：描述"走 Inductor autotune 路径时"的收益，
不是 vLLM 默认路径的收益。这句话必须写进交付文档和图注。

### bias 的处理

PR 的 `_flydsl_mxfp_bias_supported` 要求 bias 的 dtype **等于 out_dtype**、1 维、
长度为 N。vLLM 的 `apply_weights` 拿到的 bias 通常满足，但**保险起见先传 `None`、
在外面加**（如上面代码所示）。这样 bias 绝不会成为 FlyDSL 候选被拒的原因。
等主路径跑通、确认有收益之后，再试着把 bias 传进去看能否触发 epilogue 融合。

### 需要现场确认的两点 [待验证]

- **`vllm/model_executor/kernels/linear/scaled_mm/` 下已有 `pytorch.py`**，
  里面用的是 `torch._scaled_mm`（v1）。要确认它是否已有 v2 + BlockWise1x32 的分支
  可以直接复用，能复用就不必新写。
- `--linear-backend` 与 `VLLM_DISABLED_KERNELS` 两个现成开关能否直接选中新 kernel，
  能的话连环境变量都不用新加。

---

## 4. 阻碍 3：scale layout —— 实际不存在 [已验证]

最初担心的冲突是：vLLM 在 gfx950 上 swizzle MX scale，而 FlyDSL 要 NO_SWIZZLE。
**查下来这个冲突在 dense 线性层上不存在**：

- `mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, alignment=0)`
  ——**默认就是不 swizzle**，`rocm_native.py` 用的正是默认值
- 权重 scale 的形状是 `[N, K//32]`，正是契约要求的连续布局
- 我最初看到的 `should_use_cdna4_mx_scale_swizzle()` 位于 `mxfp4_utils.py`，
  服务的是 **MoE** 路径（`aiter_mxfp4_w4a8_moe`）。**我们三个模型都是 dense，不走那条路。**

唯一要做的是**一次位转换**：scale 以 `torch.uint8` 存储
（`MXFP8_SCALE_DTYPE = torch.uint8`），而契约要求 `torch.float8_e8m0fnu`，
`.view(torch.float8_e8m0fnu)` 即可，零成本。

**但仍要在 route smoke 阶段实测验证**：打印实际传给 `_scaled_mm_v2` 的
scale dtype、形状、是否连续，与契约逐项比对。这一项不做，出问题时会表现为
"FlyDSL 候选为 0"而无从定位。

---

## 5. quark 的模拟量化陷阱 —— 分析与解法 [已验证]

`quark_ocp_mx.py` 里有一段告警：

> 平台支持 MX，但 `input_dtype != "mxfp4" or weight_dtype != "mxfp4"` 时，
> "Simulated weight dequantization and activation QDQ will be used,
> with the linear layers computed in high precision."

**这是本轮最危险的陷阱**：落进去测到的是"反量化 + bf16 GEMM"的开销，
**结果会比 bf16 还慢**，却看起来像是一组正常的 mxfp8 数据。

### 但它的适用范围比最初判断的窄

那段逻辑属于 **quark scheme**。kernel 注册表里 ROCm 的 mxfp8 有原生实现
`RocmDotScaledMxfp8LinearKernel`（注释明写 "Native CDNA4 (gfx950) MX linear"）。
所以 mxfp8 并非必然走模拟——取决于走哪条 scheme。

### 解法：三道锁，互相独立

1. **绕开 quark**：采用 §3 的新 kernel 类并置于候选首位，
   `apply_weights` 由我们自己实现，不经过 `quark_ocp_mx.ocp_mx_linear`。
2. **禁用 emulation 兜底**：用现成的
   `VLLM_DISABLED_KERNELS=EmulationMxfp8LinearKernel,EmulationMxfp4LinearKernel`。
   这样一旦我们的 kernel 不可用，服务会**直接报错而不是静默退化**。
   这是三道锁里最重要的一道——**把静默失败变成显式失败**。
3. **日志断言**：服务日志里**不得出现** `Simulated weight dequantization`
   或 `computed in high precision`。写进 route smoke 的验收，命中即判该点位失败。

---

## 6. 测试矩阵（与 bf16 轮次一致）

| 维度 | 取值 | 点数 |
|---|---|---|
| 模型 | Llama-3.1-8B-Instruct、Qwen3-32B、Llama-3.3-70B-Instruct | 3 |
| 精度 | mxfp8、mxfp4（**分两轮跑，各自独立的 run 与交付目录**） | 2 |
| 工作负载 | showcase = ISL 256 / OSL 512 | 1 |
| 并发 | 8, 16, 32, 64, 128, 256 | 6 |
| 臂 | baseline `ATEN,TRITON,CK` / treatment `+FLYDSL`（实质是 ATen vs FlyDSL，见 §1） | 2 |
| 重复 | n ≥ 2，ABBA 交错，两臂背靠背同卡 | 2 |

每种精度 3 × 6 × 2 × 2 = **72 个点**，两种精度共 144 个点。

---

## 7. 固定配置

与 bf16 轮次**完全一致**的部分（不要改，否则三种精度无法横向比较）：

| 参数 | 值 |
|---|---|
| `E2E_TP` | 1 |
| `E2E_PROFILE_showcase` | `256:512` |
| `E2E_CONCURRENCY_SWEEP` | `8 16 32 64 128 256` |
| `E2E_MAX_NUM_BATCHED_TOKENS` | 2048 |
| `E2E_MAX_MODEL_LEN` | 1024 |
| `E2E_MAX_NUM_SEQS` | 256 |
| `E2E_REPEATS` | 2 |
| `E2E_AUTOTUNE_SEARCH_SPACE` | EXHAUSTIVE |
| `E2E_FLYDSL_AUTOTUNING` | 1 |
| 客户端 | `vllm bench serve`，prefix caching 关闭，`ignore_eos`，seed 0 |

mxfp 专用项：

| 参数 | 值 | 作用 |
|---|---|---|
| **`E2E_PRECISION`** | `mxfp8` / `mxfp4` | **一个变量切换 venv、torch 源码树、KV pin、交付目录** |
| `E2E_QUANT` | `mxfp8` / `mxfp4` | 传给 vLLM 的 `--quantization` |
| `VLLM_FORCE_ATEN_SCALED_MM` | 1 | §3 的开关，**两臂都开** |
| `VLLM_DISABLED_KERNELS` | `EmulationMxfp8LinearKernel,EmulationMxfp4LinearKernel` | §5 第 2 道锁 |
| `E2E_BACKENDS_baseline` | `ATEN,TRITON,CK` | |
| `E2E_BACKENDS_treatment` | `ATEN,TRITON,CK,FLYDSL` | |

`E2E_TRITON_DEFAULT_SPACE` 在 mxfp 轮次**无意义**（Triton 没有 MX lowering），
保留设置但不指望它起作用，并在 provenance 里如实记录。

### 隔离机制 [已验证]

```
E2E_PRECISION=bf16   → .venv-e2e   pytorch_e2e    state/kv_pin.d/bf16   bf16_result
E2E_PRECISION=mxfp8  → .venv-mxfp  pytorch_mxfp   state/kv_pin.d/mxfp8  mxfp8_result
E2E_PRECISION=mxfp4  → .venv-mxfp  pytorch_mxfp   state/kv_pin.d/mxfp4  mxfp4_result
```

bf16 的三个 KV pin 已迁入 `state/kv_pin.d/bf16/`，
`calibrate_kv_all.sh` 只删本次标定的模型的 pin。**bf16 已发布的数据与环境不可能被覆盖。**

### KV 要重新标定

量化后权重大幅缩小，KV 可用空间随之变大，**每种精度、每个模型都要重新标定**：

```bash
E2E_PRECISION=mxfp8 E2E_QUANT=mxfp8 bash scripts/bench/calibrate_kv_all.sh
```

**标定必须等完全结束再启动 sweep。** bf16 轮次踩过：标定写了两次 pin，
先启动的分片用旧值、后排队的用新值，造成 r1/r2 之间 KV 差 0.44%。
判据是标定进程数归零，不是日志里出现一次 pin 值。

---

## 8. 预期收益

kernel 层数据来自 PR #196719 的 comment（MI355X，对比 ATen）：

| 精度 | 几何平均 | 小 M（decode 区） | 大 M（prefill 区） |
|---|---|---|---|
| MXFP8 | **1.58×** | M=32 时 1.50–2.23× | M=2048–4096 时 1.38–1.60× |
| MXFP4 | **1.68×** | M=32 时 2.38–2.96× | M=2048–4096 时 1.35–1.48× |

比 bf16 轮次的 kernel 增益（decode 形状中位 +0.5%~+2.9%）**大一个数量级**。

**但不要直接把 1.58×/1.68× 当成端到端预期。** bf16 轮次已量化过转化损耗：
Llama-3.3-70B 的 decode GEMM 台面增益 1.5%–11.2%，**只有 10–20% 兑现到 TPOT**，
因为它贴着 HBM 天花板；Llama-3.1-8B 带宽利用率仅 23–39%，几乎全额兑现。

**量化会改变这个图景**——这是本轮最值得验证的假设：

| 模型 | bf16 权重 / decode 带宽下限 | mxfp8 | mxfp4 |
|---|---|---|---|
| Llama-3.1-8B | 14.0 GB / 2.24 ms | ~7 GB / 1.12 ms | ~3.5 GB / 0.56 ms |
| Qwen3-32B | 58.4 GB / 9.34 ms | ~29 GB / 4.67 ms | ~15 GB / 2.34 ms |
| Llama-3.3-70B | 136.9 GB / 21.90 ms | ~68 GB / 10.95 ms | ~34 GB / 5.48 ms |

**Llama-3.3-70B 在 bf16 下 100% 贴天花板（实测 decode 线性层 21.74 ms vs 下限
21.73 ms），mxfp4 把权重流量砍到四分之一后应当出现明显余量。**
如果它在 mxfp4 上终于显出收益，就直接验证了 bf16 轮次得出的那条规律。

---

## 9. 执行步骤

```bash
# ---- 前置，必须串行，每步都要验收 ----
# 0a. 独立 worktree + 独立 venv 装 PR 分支（§2），用验收脚本确认
# 0b. 微基准冒烟：mxfp8 与 mxfp4 各跑一次，确认出现 flydsl 候选（§2）
# 0c. 打 vLLM 补丁（§3），两臂都生效
# 0d. 用小模型（Qwen3-0.6B）打通整条证据链（§10），再上三个大模型
# 0e. 量化权重（§11）
# 0f. KV 标定，等进程归零再往下

# ---- 本体，每种精度独立一轮 ----
for Q in mxfp8 mxfp4; do
  E2E_PRECISION=$Q E2E_QUANT=$Q \
  VLLM_FORCE_ATEN_SCALED_MM=1 \
  VLLM_DISABLED_KERNELS=EmulationMxfp8LinearKernel,EmulationMxfp4LinearKernel \
  E2E_BACKENDS_baseline=ATEN,TRITON,CK \
  E2E_BACKENDS_treatment=ATEN,TRITON,CK,FLYDSL \
    bash scripts/run_all.sh
  bash scripts/supervisor.sh &
done
```

出图、发布、合并图与 bf16 轮次完全一致（stage 08/09/10）。
`combined.py` 的三模型配色与纵轴自动缩放已就位，**无需改动**。

---

## 10. 验收：怎么证明真的测到了 mxfp

bf16 轮次踩过的坑在这里只会更多（多了量化、scale layout、打包格式三层）。
**每一条都要有证据，缺一条就不能发布数据。**

| # | 要证明什么 | 怎么证明 | 失败的样子 |
|---|---|---|---|
| 1 | 模型真的被量化了 | 权重显存占用应约为 bf16 的 1/2（mxfp8）或 1/4（mxfp4） | 仍是 bf16，测的是 bf16 |
| 2 | **没有落进模拟量化** | 日志里**不得出现** `Simulated weight dequantization` / `computed in high precision`；并用 `VLLM_DISABLED_KERNELS` 把兜底变成硬失败 | 测到反量化开销，**比 bf16 还慢** |
| 3 | 线性层变成了 `_scaled_mm_v2` | Inductor 日志出现 `AUTOTUNE _scaled_mm_v2(...)` | 只有 `AUTOTUNE mm(`，补丁没生效 |
| 4 | 契约逐项满足 | 打印实际的 scale dtype / 形状 / 连续性 / recipe / swizzle，与 §1 比对 | 任一项不符 → FlyDSL 候选为 0，且**静默** |
| 5 | FlyDSL 产生了候选 | autotune 块里出现 `flydsl_*` | 候选为 0 |
| 6 | FlyDSL 赢了 | 胜出者是 `dtypes:` 行之后的**第一条**候选 | 取错行会得出"从未胜出"的假结论（bf16 轮次真的踩过） |
| 7 | 胜出的 kernel 真在跑 | 生成代码里的 `async_compile.flydsl().run()` | 踩过 kernelName 解析失败静默回落的坑 |
| 8 | 数值正确 | 小规模精度校验，见下 | scale layout 错会表现为全错或 NaN，而非正常量化误差 |

**数值校验的判读**：mxfp4 只有 2 位尾数，困惑度相对 bf16 上升是正常的；
但若输出为 NaN、全零，或与 bf16 参考的相对误差达到 O(1)，
那是 **scale layout 或打包格式错了**，不是量化误差。
建议先用 Qwen3-0.6B 跑通这一条，再上三个大模型。

### route smoke 与 route evidence 要扩展

现有脚本只认 `mm` 的 autotune 块和 `flydsl_mm_*` 符号。mxfp 轮次要同时认
`_scaled_mm_v2` 块与对应符号，**并且必须先过阳性对照**——用一个已知会命中的形状
验证检测器本身有效，否则"没检测到"和"检测器坏了"分不清。
bf16 轮次在这一点上吃过亏。

---

## 11. 量化权重从哪来 [待验证]

**这是方案里最大的未知数，必须开跑前定下来。** 三条路：

1. **已量化的 checkpoint**：查 AMD 用 Quark 发布的 OCP-MX 版本。省时，
   但量化配方（block size、scale dtype、打包方式）不一定匹配我们的契约。
2. **加载时在线量化**：`quark_ocp_mx.py` 有 `dynamic_mxfp4_quant`，
   看起来支持从 bf16 权重在线量化成 mxfp4；**mxfp8 是否同样支持要确认**。
   不需要额外 checkpoint，但每次启动都要量化一遍，要计入时间预估。
3. **离线量化**：用 Quark 工具链自己量化并落盘。最可控，最费时。

**三个模型、两种精度共 6 份权重必须同一套工具、同一套配方**，
否则模型之间不可比。

---

## 12. 已知风险与预案

| 风险 | 预案 |
|---|---|
| 契约某一项不满足导致 FlyDSL 候选为 0，且静默 | §10 第 4 条：把实际传参打印出来逐项比对，不要只看"有没有候选" |
| 落进模拟量化却不自知 | §5 三道锁，尤其 `VLLM_DISABLED_KERNELS` 把静默退化变成硬失败 |
| 新 kernel 类让 baseline 也偏离 vLLM 默认路径 | 两臂都用同一条路径，差异仍只有 FLYDSL 一项；但结论的条件性要写进图注 |
| 量化权重来源不一致 | 6 份权重同工具同配方 |
| KV pin / 数据被跨精度覆盖 | **已解决**：`E2E_PRECISION` 隔离 venv、源码树、pin、结果目录 [已验证] |
| 标定未结束就启动 sweep | 判据改为"标定进程数归零"，不是日志出现 pin 值 |
| 编译时间爆炸 | scaled GEMM 候选空间可能比 bf16 大；先在单点位测一次单点耗时再排全矩阵 |
| mxfp4 数值误差影响 token 数 | `ignore_eos` 固定输出长度，token 数不受质量影响；仍记录困惑度作为佐证 |

---

## 13. 时间预估

| 阶段 | 预计 |
|---|---|
| worktree + 独立 venv 构建 PyTorch 并验收 | **3–5 小时**（全量编译） |
| 微基准冒烟（mxfp8 + mxfp4） | 0.5 小时 |
| vLLM 补丁 + 小模型打通证据链 | 2–4 小时 |
| 量化权重（6 份，取决于来源） | 1–6 小时 |
| KV 标定（3 模型 × 2 精度） | 1 小时 |
| **benchmark 本体**（每精度 36 分片，8 卡并行） | 每精度 2–3 小时，共 **4–6 小时** |

**总计 1.5–2 个工作日**，其中绝大部分是前置打通，不是跑数据。

---

## 14. 与 bf16 轮次的对照表

| 项 | bf16 | mxfp8 / mxfp4 |
|---|---|---|
| 被测算子 | `mm` | `_scaled_mm_v2` |
| 可见性前提 | `VLLM_FORCE_ATEN_LINEAR=1` | `VLLM_FORCE_ATEN_SCALED_MM=1`（**待实现**，§3） |
| 真实对照 | ATen vs Triton vs FlyDSL | **ATen vs FlyDSL**（Triton/CK 无 MX lowering） |
| PyTorch | `pytorch_e2e` @ 313e0fb + `.venv-e2e` | `pytorch_mxfp` @ PR 分支 + `.venv-mxfp` |
| 额外契约 | 无 | scale dtype / block size / NO_SWIZZLE / 打包格式 / bias dtype |
| 权重来源 | 原始 checkpoint | 需量化（**来源待定**，§11） |
| kernel 层增益 | decode 中位 +0.5%~+2.9% | **1.58× / 1.68×（几何平均）** |
| 额外失败模式 | — | 模拟量化静默回落、scale layout 错、打包格式错、契约任一项不符 |
| 模型 / 并发 / ISL / OSL / 重复 / 出图 | — | **完全一致** |
