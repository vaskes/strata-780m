#!/usr/bin/env python3
"""Find the largest prompt this 780M build actually completes, and measure decode cleanly.

prefill on long prompts dies in rocBLAS on gfx1103 (gfx1103 is not in the official ROCm
rocBLAS arch list), so this walks short->long and stops at the first failure.
"""
import json, re, subprocess, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "vaskes-strata-780m"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
LOG = "/data/state/strata/launch/docker.engine.log"


def ask(n, gen=256):
    prompt = ("The strata engine streams experts from a resident cache and falls back to the pack "
              "on disk when a rank misses. ") * max(1, n // 15) + "\n\nSummarise in one sentence."
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": gen, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
        return (time.time() - t0), out, None
    except Exception as e:
        return (time.time() - t0), None, str(e)


def last_line():
    txt = subprocess.run(["docker", "exec", "strata-780m", "tail", "-n", "400", LOG],
                         capture_output=True, text=True, timeout=60).stdout
    best = None
    for line in txt.splitlines():
        m = re.search(r"prompt (\d+) tokens = .*?read in ([\d.]+) ms \(([\d.]+) tok/s\)"
                      r".*?(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)", line)
        if m:
            best = m
    if not best:
        return None
    return {"prompt_tokens": int(best.group(1)), "prompt_ms": float(best.group(2)),
            "prompt_tps": float(best.group(3)), "generated": int(best.group(4)),
            "decode_ms": float(best.group(5)), "decode_tps": float(best.group(6))}


print("warming up...", flush=True)
ask(32, 32)
print("done\n", flush=True)

rows, last_ok = [], 0
for n in (64, 128, 256, 512, 1024, 2048, 4096):
    wall, out, err = ask(n)
    if err:
        print(f"  ~{n:>5} tok prompt : FAILED after {wall:5.1f}s  ({err})", flush=True)
        break
    time.sleep(2)
    s = last_line()
    row = {"approx_prompt": n, "wall_s": round(wall, 1),
           "server_prompt_tokens": s["prompt_tokens"] if s else None,
           "prompt_tps": s["prompt_tps"] if s else None,
           "decode_tps": s["decode_tps"] if s else None,
           "ttft_s": round(s["prompt_ms"] / 1000, 1) if s else None}
    rows.append(row)
    last_ok = n
    print(f"  ~{n:>5} tok prompt : prefill {s['prompt_tps']:6.1f} tok/s   decode {s['decode_tps']:5.1f} tok/s"
          f"   TTFT {s['prompt_ms']/1000:5.1f}s   wall {wall:6.1f}s", flush=True)

print(f"\nlargest prompt that completed: ~{last_ok} tokens")
json.dump(rows, open("/opt/strata-780m/bench-780m.json", "w"), indent=1)