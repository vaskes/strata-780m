# GRUB parameters — the part that actually makes this work

**This is the main blocker.** Everything else in this repo — the Docker build, the
gfx1103 packaging, the tuning flags — is downstream of getting these kernel parameters
right. Skip them and nothing runs at all, or hangs in ways that look like driver bugs.

Read this before you try anything else.

---

## TL;DR

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt ttm.pages_limit=13369344 ttm.page_pool_size=8021606 amdgpu.gttsize=52224 amdgpu.lockup_timeout=60000 amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 transparent_hugepage=always"
```

If you read nothing else: **`amdgpu.cwsr_enable=0` is the single parameter that makes
this possible.** Everything else sizes and tunes. Without CWSR off, the GPU hangs after
5–30 seconds of any nontrivial inference and no amount of engine tuning will save it.

---

## Two separate jobs

The parameters fall into two groups that are easy to confuse:

1. **Does it run at all?** — `amdgpu.cwsr_enable=0` and friends. Missing these means
   nothing works, regardless of hardware.
2. **How much memory does the GPU get?** — `amdgpu.gttsize`, `ttm.pages_limit`,
   `ttm.page_pool_size`. These decide whether a model fits, and they must be sized for
   *your* RAM.

---

## Group 1 — making it run (mandatory)

| Parameter | Mandatory | What it does |
|---|:---:|---|
| `amdgpu.cwsr_enable=0` | **yes** | **THE FIX.** Disables compute-wave-save-restore. On gfx1103, CWSR triggers a MES firmware bug that hangs the GPU after 5–30 s of real work. ROCm issues #5590, #5665. |
| `amdgpu.mes_kiq=1` | yes | Use the KIQ (Kernel Interface Queue) for the MES scheduler. Required on gfx1103; avoids the KFD queue-eviction hang. |
| `amd_iommu=on` | yes | **Do not set this to `off`.** IOMMU is stability-critical here. |
| `iommu=pt` | yes | Pass-through mode — required with IOMMU on, and faster. |
| `amdgpu.gpu_recovery=1` | recommended | Try to recover the GPU on a hang instead of leaving it dead. Useful while debugging. |
| `amdgpu.lockup_timeout=60000` | recommended | 60 s before a lockup is declared. Lower values abort runs that were merely slow. |
| `amdgpu.sg_display=0` | recommended | No scatter-gather display path; frees that memory for compute. |
| `amdgpu.noretry=1` | optional | Fail fast instead of retrying a bad command. |
| `transparent_hugepage=always` | recommended | THP for GTT allocations, less fragmentation under large-tensor load. |

## Group 2 — sizing the memory

| Parameter | What it is |
|---|---|
| `amdgpu.gttsize=<MiB>` | GTT (Graphics Translation Table) size, in MiB. A **hard cap** the driver applies at init. On an APU this is the bulk of your GPU-visible memory. |
| `ttm.pages_limit=<pages>` | Hard cap on TTM pages (4 KiB each) the kernel may allocate globally. **Must be ≥ gttsize**, or torch sees a truncated GTT and you OOM on a model that should fit. |
| `ttm.page_pool_size=<pages>` | Initial free-page pool for fast allocation. A performance hint, not a cap. ~60% of `pages_limit` suits large tensors. |

On an APU the BIOS reserves a fixed VRAM carveout (**4 GiB** on this board, verified via
`mem_info_vram_total`) and GTT sits on top. Total GPU-addressable memory is therefore:

```
total = VRAM carveout (4 GiB, fixed in BIOS) + amdgpu.gttsize
```

On this host: `4096 MiB VRAM + 52224 MiB GTT = 56320 MiB (55 GiB)` out of 58.2 GiB of
RAM. The remaining ~3 GiB goes to the kernel and the CPU side.

---

## The arithmetic

With 4 KiB pages (check yours: `getconf PAGESIZE`):

```
pages = MiB × 256          1 MiB = 256 pages
                         1 GiB = 262144 pages
```

So for any target GTT size:

```bash
NEW_GTT_MiB=52224
NEW_PAGES=$(( NEW_GTT_MiB * 256 ))          # ttm.pages_limit
NEW_POOL=$(( NEW_PAGES * 60 / 100 ))        # ttm.page_pool_size
```

Give `pages_limit` a little headroom over `gttsize` if you can spare it — TTM also
services VRAM and system pages. Both configs below do that:

| GTT (`gttsize`) | `pages_limit` | headroom | `page_pool_size` |
|---:|---:|---:|---:|
| 40,960 MiB (40 GiB) | 10,485,760 | 0 | 6,291,456 |
| 52,224 MiB (51 GiB) | 13,369,344 | 0 | 8,021,606 |
| 92,160 MiB (90 GiB) | 25,165,824 | 6 GiB | 15,099,494 |
| 108,544 MiB (106 GiB) | 28,311,552 | 2 GiB | 16,986,931 |

**Rule: `ttm.pages_limit × 4 KiB ≥ amdgpu.gttsize`.** If it is smaller, the kernel cannot
back the full GTT and tensors get evicted or OOM'd despite `free` memory looking fine.

---

## Ready-to-paste configs

Pick by your total system RAM. The model has to fit in `4 GiB + GTT`, with a few GiB left
for the kernel.

### 64 GB class, 4-bit quant (IQ4_XS / Q4_K_M) — this is the verified one

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt ttm.pages_limit=13369344 ttm.page_pool_size=8021606 amdgpu.gttsize=52224 amdgpu.lockup_timeout=60000 amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 transparent_hugepage=always"
```

55 GiB GPU-addressable. Measured working on llmhost2 with `Qwen3.8-Flash-Next UD-IQ4_XS`.

### 128 GB class, Q8_0 — the `#qwen 177b` line

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt ttm.pages_limit=28311552 amdgpu.gttsize=108544 amdgpu.lockup_timeout=60000 amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 transparent_hugepage=always"
```

110 GiB GPU-addressable. This line sits commented out on the current 64 GB host because
it needs a **128 GB** box — a Q8_0 177B pack is roughly twice the bytes of the IQ4_XS one,
and the active config has no room for it. Note it also lacks the cwsr/mes group: it was
written before that fix was found. **If you use the 128 GB config, add the cwsr group to
it or you will reintroduce the hang.**

### 96 GB GTT variant, also seen on this host

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=on iommu=pt ttm.pages_limit=25165824 amdgpu.gttsize=92160 amdgpu.lockup_timeout=60000 amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 transparent_hugepage=always"
```

---

## Applying

```bash
# 1. Back up
sudo cp /etc/default/grub /etc/default/grub.bak.$(date +%s)
sudo cat /proc/cmdline > /root/cmdline.before.txt

# 2. Edit
sudo nano /etc/default/grub        # set GRUB_CMDLINE_LINUX_DEFAULT

# 3. Apply
sudo update-grub
sudo reboot

# 4. Verify after reboot
cat /proc/cmdline | tr ' ' '\n' | grep -E 'amdgpu|ttm|iommu'
echo "GTT:          $(( $(cat /sys/class/drm/card0/device/mem_info_gtt_total)   / 1024 / 1024 )) MiB"
echo "VRAM:         $(( $(cat /sys/class/drm/card0/device/mem_info_vram_total)  / 1024 / 1024 )) MiB"
echo "pages_limit:  $(( $(cat /sys/module/ttm/parameters/pages_limit) * 4 / 1024 )) MiB"
echo "page_pool:    $(( $(cat /sys/module/ttm/parameters/page_pool_size) * 4 / 1024 )) MiB"
for p in cwsr_enable mes_kiq noretry sg_display gpu_recovery; do
  echo "amdgpu.$p = $(cat /sys/module/amdgpu/parameters/$p)"
done
```

The `card0`/`card1` index varies — find yours with
`ls /sys/class/drm/card*/device/mem_info_gtt_total`.

### Rollback

```bash
sudo cp /etc/default/grub.bak.<timestamp> /etc/default/grub
sudo update-grub && sudo reboot
```

---

## Symptoms of wrong values

| Symptom | Cause | Fix |
|---|---|---|
| GPU hangs 5–30 s into any real inference | `amdgpu.cwsr_enable` not disabled | **THE FIX** — set it to 0 |
| torch sees far less memory than GRUB requests | `ttm.pages_limit` < `amdgpu.gttsize` | raise `pages_limit` |
| OOM on a model that should fit | `page_pool_size` too small; TTM evicts constantly | raise to ~60% of `pages_limit` |
| `dmesg: ttm: Maximum number of pages exceeded` | `pages_limit` too low | raise it |
| `dmesg: amdgpu: Not enough memory for GTT` | `gttsize` exceeds physical RAM | lower `gttsize` |
| GTT unchanged after editing GRUB | forgot `update-grub`, or no reboot | both, then re-check sysfs |
| Long stalls, model crawls, huge `used` vs `RSS` | GTT thrashing — pool too small | raise `page_pool_size` |

If RAM reads as "used" with nothing resident, look at `/proc/meminfo` field **`GPUActive`**
(the amdgpu TTM pool), not free/available. A large `GPUActive` is expected and healthy.

---

## What is deliberately NOT changed

* **`amd_iommu=off`** — stability-critical. Do not disable IOMMU.
* **Kernel version** — no upgrades or downgrades; these parameters are validated against
  the 7.0.0-generic line.
* **`/etc/modprobe.d/`** — can conflict with the GRUB cmdline. Leave it alone.
* **BIOS** — the 4 GiB VRAM carveout and GTT are set correctly for this use. The amd\_iommu
  BIOS option must match `amd_iommu=on`, or the cmdline is ignored for IOMMU purposes.

---

## Cross-reference

* `docs/ENVIRONMENT.md` — full hardware and kernel state of the measured host.
* `docs/TTM_GPU_MEMORY.md` — deeper GTT sizing reference and symptom table.
* `comfyui-rocm-780m` repo, `grub-backup/` — the original before/after capture of this
  change, with per-parameter sources.
* `docs/REPORT.md` — the full investigation, including why the 177B Q8_0 config is
  commented out on the current host.