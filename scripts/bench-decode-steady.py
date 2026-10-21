#!/usr/bin/env python3
"""Steady-state decode on the 780M: 256 generated tokens, short prompt, three runs.

29-token generations are mostly ramp (first token, then MTP has to warm). This measures
the number that actually matters for typing speed, and repeats it so the spread is visible.
"""
import json, re, subprocess, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "vaskes-strata-780m"
MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
LOG = "/data/state/strata/launch/docker.engine.log"
PROMPT = "Explain how a mixture-of-experts model routes tokens. "


def ask(gen):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": gen, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return time.time() - t0, out


def last_server_line():
    txt = subprocess.run(["docker", "exec", "strata-780m", "tail", "-n", "600", LOG],
                         capture_output=True, text=True, timeout=60).stdout
    best = None
    for line in txt.splitlines():
        m = re.search(r"prompt (\d+) tokens = .*?read in ([\d.]+) ms \(([\d.]+) tok/s\)"
                      r".*?(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)"
                      r".*?drafts accepted (\d+) of (\d+)", line)
        if m:
            best = m
    if not best:
        return None
    return dict(ptok=int(best.group(1)), p_ms=float(best.group(2)), p_tps=float(best.group(3)),
                gen=int(best.group(4)), d_ms=float(best.group(5)), d_tps=float(best.group(6)),
                acc_a=int(best.group(7)), acc_b=int(best.group(8)))


print("warmup...", flush=True)
ask(32)
time.sleep(3)

rows = []
for i in range(3):
    wall, out = ask(256)
    time.sleep(3)
    s = last_server_line()
    if not s:
        print(f"run {i+1}: no server timing", flush=True)
        continue
    rows.append(s)
    acc = f"{s['acc_a']}/{s['acc_b']}" + (f" ({100*s['acc_a']/s['acc_b']:.0f}%)" if s['acc_b'] else "")
    print(f"run {i+1}: prompt {s['ptok']:>4} tok @ {s['p_tps']:5.1f} t/s | "
          f"decode {s['gen']} tok @ {s['d_tps']:5.2f} t/s ({s['d_ms']/1000:5.1f}s) | "
          f"drafts {acc} | wall {wall:5.1f}s", flush=True)

if rows:
    d = sorted(r["d_tps"] for r in rows)
    print(f"\nbest {max(d):.2f} tok/s | median {d[len(d)//2]:.2f} tok/s | worst {min(d):.2f} tok/s")
    acc = [100*r["acc_a"]/r["acc_b"] for r in rows if r["acc_b"]]
    if acc:
        print(f"draft acceptance: {sum(acc)/len(acc):.0f}%")
    json.dump(rows, open("/opt/strata-780m/decode-steady.json", "w"), indent=1)