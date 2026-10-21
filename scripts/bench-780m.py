#!/usr/bin/env python3
"""Strata on Radeon 780M (gfx1103) — prompt / decode / TTFT measurement.

Drives the running server through its OpenAI-compatible API and reads the server-side
timings the engine logs (the same source bench/bench_vs_llama.py uses), because a
client-side stopwatch cannot separate prompt processing from token generation.

Cells match docs/RADEON_780M.md §7 so the numbers are comparable with the community
run on a Ryzen 7 8745HS: 4K and 32K prompts, 256 generated tokens, greedy decoding.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "vaskes-strata-780m"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
GEN = 256

LOG = "/data/state/strata/launch/docker.engine.log"


def ask(prompt, gen=GEN, temp=0.0):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": temp,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return time.time() - t0, out


def approx_tokens(text):
    """~4 chars/token for English prose, close enough to hit a cell size."""
    return len(text) / 4


def make_prompt(n_tokens, seed):
    # deterministic filler, so every cell gets the same kind of text
    para = ("The strata engine streams experts from a resident cache and falls back to the "
            "pack on disk when a rank misses. Prompt work runs on matrix cores while decode "
            "runs on the packed vector path, so the two phases scale differently with length. ")
    text = (para * ((n_tokens * 4) // len(para) + 2))[: n_tokens * 4]
    return f"[case {seed}, ~{n_tokens} tokens]\n{text}\n\nSummarise the paragraph above in one sentence."


def engine_tail(lines=200):
    try:
        return subprocess.run(
            ["docker", "exec", "strata-780m", "tail", "-n", str(lines), LOG],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


PROMPT_RE = re.compile(r"prompt[_ ](?:processed|eval)[: ]+([\d.]+)\s*tokens?\s*(?:in|at)\s*([\d.]+)", re.I)
DECODE_RE = re.compile(r"([\d.]+)\s*tokens/s", re.I)


def parse_engine(text):
    """Pull the last prompt-rate and decode-rate the engine reported.

    The engine logs one line per request, e.g.
      strata serve: prompt 4096 tokens = 0 reused + 4096 read in 15233 ms (268.9 tok/s),
                   256 generated in 1612 ms (158.8 tok/s), drafts accepted 137 of 256, ...
    """
    out = {}
    for line in text.splitlines():
        if "strata serve:" not in line or "prompt" not in line:
            continue
        # only the last complete line (partial writes at the tail are possible)
        m = re.search(r"prompt (\d+) tokens = .*?read in ([\d.]+) ms \(([\d.]+) tok/s\)"
                      r".*?(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)", line)
        if m:
            out["prompt_tokens_server"] = int(m.group(1))
            out["prompt_ms"] = float(m.group(2))
            out["prompt_tps"] = float(m.group(3))
            out["generated"] = int(m.group(4))
            out["decode_ms"] = float(m.group(5))
            out["decode_tps"] = float(m.group(6))
        d = re.search(r"drafts accepted (\d+) of (\d+)", line)
        if d:
            a, b = int(d.group(1)), int(d.group(2))
            out["draft_accept"] = f"{a}/{b}" + (f" ({100*a/b:.0f}%)" if b else "")
    return out


def main():
    print("warming up (first request is cold: profile, HIP module load, first expert fetches)...", flush=True)
    ask("Say hello.", 32, 0.0)

    results = []
    for cells, temp in (([(4000, "4k")], 0.0), ([(32000, "32k")], 0.0)):
        for n, name in cells:
            prompt = make_prompt(n, name)
            real = approx_tokens(prompt)
            print(f"\n=== cell {name} (approx {real:.0f} prompt tokens), {GEN} generated, temp {temp} ===", flush=True)
            wall, out = ask(prompt, GEN, temp)
            usage = out.get("usage", {})
            comp = usage.get("completion_tokens")
            ptok = usage.get("prompt_tokens")
            txt = out["choices"][0]["message"].get("content", "")
            eng = parse_engine(engine_tail())
            row = {
                "cell": name,
                "prompt_tokens": ptok,
                "generated": comp,
                "wall_s": round(wall, 2),
                "client_decode_tps": round(comp / wall, 2) if (comp and wall) else None,
                "prompt_tps": eng.get("prompt_tps"),
                "decode_tps": eng.get("decode_tps"),
                "answer_head": txt[:90].replace("\n", " "),
            }
            results.append(row)
            print(json.dumps(row, indent=1, ensure_ascii=False), flush=True)

    print("\n\n===== SUMMARY: Strata on Radeon 780M (gfx1103), Qwen3.8-Flash-Next UD-IQ4_XS, 32K ctx =====")
    hdr = f"{'cell':>5} {'prompt tok':>10} {'prompt t/s':>10} {'decode t/s':>10} {'TTFT':>9} {'accept':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ttft = (r["prompt_ms"] / 1000) if r.get("prompt_ms") else None
        print(f"{r['cell']:>5} {r.get('prompt_tokens_server') or 0:>10} "
              f"{str(r.get('prompt_tps') or '-'):>10} {str(r.get('decode_tps') or '-'):>10} "
              f"{(f'{ttft:.1f}s' if ttft else '-'):>9} {str(r.get('draft_accept') or '-'):>9}")
    with open("/opt/strata-780m/bench-780m.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()