import os
from evaluation import evaluate

MODELS = [
    "google/gemma-2-2b-it",
 #   "mistralai/Mistral-7B-Instruct-v0.2",
]

TOPOLOGIES = ["star", "chain"]

def main():
    os.makedirs("outputs_min", exist_ok=True)

    episodes = 5
    n_segments = 8  
    key_index = 3
    template_slot = "prefix"

    # baseline vs biased routing
    settings = [
        ("baseline", 0.0),
        ("biased", 0.8),
    ]

    rows = []
    for model in MODELS:
        for topo in TOPOLOGIES:
            for tag, rb in settings:
                cfg = {
                    "topology": topo,
                    "route_bias": rb,
                    "account_affinity": 0.20,
                    "key_index": key_index,
                    "template_slot": template_slot,
                    "max_new_tokens": 32,
                    "opt_level": tag,  
                }
                log_path = f"outputs_min/point1_{model.split('/')[-1]}_{topo}_{tag}.jsonl"
                stats = evaluate(
                    cfg,
                    episodes=episodes,
                    model_name=model,
                    n_segments=n_segments,
                    log_path=log_path,
                    label=f"{model} | {topo} | {tag}",
                )
                rows.append((model, topo, tag, stats))

    print("\n=== Minimal Point-1 Table (ASR + FA) ===")
    print("model\ttopo\tsetting\tclean\tkey_only\ttemplate_only\tboth\tFA")
    for model, topo, tag, st in rows:
        print(
            f"{model.split('/')[-1]}\t{topo}\t{tag}\t"
            f"{st['clean']:.3f}\t{st['key_only']:.3f}\t{st['template_only']:.3f}\t{st['both']:.3f}\t{st['FA']:.3f}"
        )

if __name__ == "__main__":
    main()
