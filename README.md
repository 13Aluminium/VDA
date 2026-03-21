# Memory-Centric Architectures for Large Language Models
**Team 12 — VDA Presentation**  
Ayush Luhar, Vatsal Patel & Dev Trivedi  
*Built on llama.cpp · Model: LLaMA-2-7B · Track B — Tiered Memory*

---

## Overview

This project proposes and validates a **3-tier memory hierarchy** for LLM inference. Instead of treating GPU HBM as the only memory tier (as systems like vLLM and llama.cpp do today), we argue that GPU memory should be treated as a *premium* tier — reserved only for the most latency-sensitive data — while cheaper, larger memory tiers absorb cold data gracefully.

The experiments in this repository were run on two real GPUs — an **NVIDIA H100 80GB HBM3** and an **NVIDIA RTX PRO 6000 Blackwell** — to validate that our findings hold across hardware generations.

---

## Repository Structure

```
├── H100.ipynb              # All experiments run on NVIDIA H100 80GB
├── RTX_PRO_6000.ipynb      # Same experiments on NVIDIA RTX PRO 6000 Blackwell
└── README.md
```

---

## The 3-Tier Architecture

| Tier | Memory Type | What Lives Here | Why |
|------|-------------|-----------------|-----|
| Tier 1 | SRAM / Scratchpad | Attention tiles, activations | Sub-nanosecond latency, used during active compute |
| Tier 2 | GPU HBM | Model weights, recent KV cache | Fast, but limited and expensive |
| Tier 3 | Host DRAM / Pooled Memory | Cold KV cache, compressed overflow | Cheap, large, absorbs what HBM cannot |

Data migrates between tiers based on recency and reuse — promoted up when needed, evicted down under pressure, and compressed before hitting Tier 3.

---

## Experiments

### Experiment 1 — Capacity Benchmark (`Slide 11`)

**What it does:**  
Loads LLaMA-2-7B in FP16 onto the GPU and runs a full prefill + single decode step across five context lengths (128 → 2048 tokens). After each run, it measures how much GPU memory the weights occupy, how much the KV cache occupies, and how long prefill and decode take.

**Technical details:**  
- Model loaded with `device_map="auto"` and `torch_dtype=torch.float16` — no quantization, clean baseline
- Memory is measured using `torch.cuda.memory_allocated()` *before* the forward pass (weights only) and *after* (weights + KV cache), giving a clean separation between the two
- KV cache size is computed by walking the `past_key_values` object returned by the model. The code handles three different cache formats that Hugging Face transformers uses depending on version: the new `DynamicCache` with `.layers`, `DynamicCache` with `.key_cache/.value_cache`, and the legacy tuple-of-tuples format
- `torch.cuda.synchronize()` is called before and after each timed section to ensure GPU operations are actually complete before the timer stops — without this, Python returns before the GPU finishes, making latency measurements meaningless
- Each sequence length gets a full `empty_cache()` + `gc.collect()` between runs to prevent memory from one run leaking into the next

**What we found:**  
Model weights are a fixed cost — they never change regardless of context length. The KV cache grows linearly at exactly 0.5 MB per token, which is a direct consequence of LLaMA-2-7B's architecture: 32 layers × 32 heads × 128 head dimension × 2 (keys + values) × 2 bytes per FP16 element. This is not a hardware property — it is a model property, and it was identical on both GPUs.

---

### Experiment 2 — Tier Transition Latency (`Slide 13`)

**What it does:**  
Physically times data moving between the three memory tiers using three separate measurement functions:
- `measure_gpu_internal_copy()` — tiles of 32–256 KB copied within GPU memory (Tier 1 ↔ Tier 2)
- `measure_gpu_to_cpu()` — blocks of 1–16 MB transferred from GPU HBM to CPU RAM (Tier 2 → Tier 3, KV eviction)
- `measure_cpu_to_gpu()` — same blocks transferred back (Tier 3 → Tier 2, KV promotion)

**Technical details:**  
- All CPU tensors are allocated with `pin_memory=True`. Pinned memory bypasses the OS virtual memory system and lives in a fixed physical address, which is required for DMA (Direct Memory Access) transfers over PCIe. Without pinning, every transfer would require an extra copy through a staging buffer, inflating latency and making bandwidth measurements unreliable
- Each function runs a warmup loop before the timed section — 10 iterations for GPU-internal copies, 5 for PCIe transfers. This is important because the first few transfers are slower due to TLB misses, IOMMU mapping overhead, and PCIe link state initialization
- Timing uses `time.perf_counter()` with `torch.cuda.synchronize()` bracketing each transfer. This gives wall-clock time that includes actual hardware transfer time, not just the time to issue the copy command
- PCIe bandwidth efficiency is calculated against the PCIe 5.0 theoretical maximum. The RTX PRO 6000 Blackwell showed significantly higher efficiency (~76%) compared to the H100 (~40%), reflecting differences in PCIe controller implementation between GPU generations
- The decode stall analysis uses the measured decode latency from Experiment 1 as the budget, then calculates what fraction of that budget a Tier 3 cache miss would consume — both with and without prefetching

**What we found:**  
Tier transitions are fast enough that they do not threaten decode throughput — *as long as prefetching is used*. Without prefetching, a cold Tier 3 miss stalls the entire decode step. With prefetching, the promotion happens in the background during preceding decode steps and the cost is fully hidden. The crossover point where KV cache size equals model weight size occurred at approximately the same token count on both GPUs, confirming this is a model property.

---

### Experiment 3 — Optimization Impact (`Slide 12`)

**What it does:**  
Measures the impact of two optimizations — KV cache compression and attention windowing — on memory footprint and bandwidth consumption.

**Technical details:**

**KV Compression:**  
The FP16 KV cache size is measured directly from the model's `past_key_values`. INT8 and INT4 sizes are computed analytically by dividing by 2 and 4 respectively — this is valid because INT4 quantization packs 4-bit values into 8-bit storage, giving a clean 4× size reduction. The code also includes a fallback path that computes KV size analytically using the model's known architecture constants (32 layers, 32 heads, 128 head dim) in case the cache format is unrecognized by the walking function.

**Attention Windowing:**  
Rather than running the model with a modified attention mask (which would require patching the model), windowing impact is computed analytically using the measured KV-per-token constant. For a given sequence length and window size, `min(seq_len, window) × KV_per_token` gives the actual memory read per decode step. The reduction percentage is `(full_attention - windowed) / full_attention`. This approach is valid because attention memory traffic scales linearly with the number of tokens attended to.

**Prefetch stall analysis:**  
Uses the `measure_tier_transfer()` function to measure actual promotion latency for a 4 MB block (a realistic KV eviction unit), then models cumulative stall cost for 1, 2, 3, 5, and 10 cold misses in a row. The "with prefetch" case is always 0ms — this is the theoretical ceiling assuming perfect prefetch prediction, which is achievable in practice with simple LRU-based lookahead.

**What we found:**  
INT4 compression delivers a consistent 4× memory reduction at every context length with no new hardware required. Attention windowing cuts memory reads dramatically at long contexts but introduces a hard tradeoff — the model loses access to context older than the window. These two optimizations are complementary: compression makes Tier 3 storage more efficient, and windowing reduces how much pressure Tier 2 faces per decode step.

---

## Hardware Used

| Property | H100 80GB HBM3 | RTX PRO 6000 Blackwell |
|----------|---------------|------------------------|
| VRAM | 79.18 GB | 94.97 GB |
| Architecture | Hopper | Blackwell |
| PyTorch | 2.10.0+cu128 | 2.10.0+cu128 |
| Model | LLaMA-2-7B FP16 | LLaMA-2-7B FP16 |

---

## Dependencies

```bash
pip install transformers accelerate safetensors
pip install bitsandbytes --prefer-binary \
  --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

> **Note:** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running to avoid CUDA OOM fragmentation errors on large sequence lengths.

---

## Key Takeaways

- **Memory pressure is a model property, not a hardware fluke.** The same KV growth rate and crossover point appeared on both GPUs.
- **Tier transitions are fast — the bottleneck is unprefetched misses.** With intelligent prefetching, tier movement cost drops to zero on the critical path.
- **INT4 compression is a free 4× capacity win.** No hardware change, no architectural modification needed.
- **Attention windowing trades recall for efficiency.** At long contexts it eliminates most memory traffic, but cold history must live in Tier 3 to remain accessible.
- **The real bottleneck is decode compute, not PCIe.** PCIe bandwidth headroom far exceeds the eviction rate needed to sustain typical decode speeds.

---

## References

- Dao et al., *FlashAttention*, NeurIPS 2022
- Dao et al., *FlashAttention-2*, arXiv 2023
- Kwon et al., *PagedAttention*, arXiv 2023
- Prabhu et al., *vAttention*, arXiv 2024
- Liu et al., *KIVI: 2-bit KV Cache Quantization*, arXiv 2024
- Xiao et al., *Attention Sinks*, arXiv 2023
- Zhang et al., *H2O*, arXiv 2023
- Sheng et al., *FlexGen*, arXiv 2023
- Rajbhandari et al., *ZeRO-Infinity*, arXiv 2021
