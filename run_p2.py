import os
import math
from evaluation import evaluate

MODELS = [
    "google/gemma-2-2b-it",
    # "mistralai/Mistral-7B-Instruct-v0.2",
]

TOPOLOGIES = ["star"]

BIAS_GRID = [0.0, 0.4]  


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def main():
    os.makedirs("outputs_min", exist_ok=True)

    episodes = 5         
    n_segments = 3       
    key_index = 0
    template_slot = "prefix"

    rows = []

    for model in MODELS:
        for topo in TOPOLOGIES:
            surrogate_list = []
            empirical_list = []

            for rb in BIAS_GRID:
                cfg = {
                    "topology": topo,
                    "route_bias": rb,
                    "account_affinity": 0.20,
                    "key_index": key_index,
                    "template_slot": template_slot,
                    "max_new_tokens": 64,
                    "opt_level": f"rb{rb}",
                }

                log_path = f"outputs_min/point2_{model.split('/')[-1]}_{topo}_rb{rb}.jsonl"
                stats = evaluate(
                    cfg,
                    episodes=episodes,
                    model_name=model,
                    n_segments=n_segments,
                    log_path=log_path,
                    label=f"{model} | {topo} | rb={rb}",
                )

                surrogate_list.append(stats["ASR_surrogate_emp"])
                empirical_list.append(stats["ASR_both_empirical"])
                rows.append((model, topo, rb, stats))

            r = pearson_r(surrogate_list, empirical_list)
            print(f"\n[fidelity correlation] model={model.split('/')[-1]} topo={topo} Pearson r = {r:.3f}")
            print("  surrogate:", [round(x, 3) for x in surrogate_list])
            print("  empirical:", [round(x, 3) for x in empirical_list])

    print("\n=== Table ===")
    print("model\ttopo\troute_bias\tclean\tkey_only\ttemplate_only\tboth\tFA\tP_route\tP_template\tsurr\temp\t|err|")
    for model, topo, rb, st in rows:
        print(
            f"{model.split('/')[-1]}\t{topo}\t{rb}\t"
            f"{st['clean']:.3f}\t{st['key_only']:.3f}\t{st['template_only']:.3f}\t{st['both']:.3f}\t{st['FA']:.3f}\t"
            f"{st['P_route_emp']:.3f}\t{st['P_template_emp']:.3f}\t{st['ASR_surrogate_emp']:.3f}\t{st['ASR_both_empirical']:.3f}\t{st['ASR_surrogate_abs_err']:.3f}"
        )


if __name__ == "__main__":
    main()
