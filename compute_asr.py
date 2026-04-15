import json, sys

p = sys.argv[1]
tot = {}
hit = {}
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        s = r["scenario"]
        tot[s] = tot.get(s, 0) + 1
        hit[s] = hit.get(s, 0) + (1 if r["activated"] else 0)

for s in ["clean","key_only","template_only","both"]:
    asr = hit.get(s,0) / max(1, tot.get(s,0))
    print(s, asr)

fa = (hit.get("key_only",0)/max(1,tot.get("key_only",0))) + (hit.get("template_only",0)/max(1,tot.get("template_only",0)))
print("FA", fa)
