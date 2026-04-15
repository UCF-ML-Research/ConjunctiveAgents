import csv
import os
import torch

from optimization import train_surrogate, TEMPLATE_SLOTS
from evaluation import evaluate

MODELS = {
    # "Gemma-2B":   "google/gemma-2-2b-it",
    # "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    # "LLaMA3-8B":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "LLaMA4-17B":  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

def build_account_positions(n_segments: int) -> torch.Tensor:
    acc = torch.zeros(n_segments)
    acc[::10] = 1.0
    return acc

def agg_min_mean_max(values):
    if not values:
        return (0.0, 0.0, 0.0)
    mn = min(values)
    mx = max(values)
    mean = sum(values) / len(values)
    return mn, mean, mx

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def format_pct(x):
    return f"{100.0*x:.1f}"

def write_table1_tex(path, grouped):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{tabular}{ll l cccc cccc ccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Topology & Opt & \multicolumn{4}{c}{Before ASR (\%)} & \multicolumn{4}{c}{After ASR (\%)} & ASR-m & ASR & ASR-M \\")
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(r" & & & Clean & Key & Tpl & Both & Clean & Key & Tpl & Both &  &  &  \\")
    lines.append(r"\midrule")

    for row in grouped:
        lines.append(
            f"{row['model']} & {row['topology']} & {row['level']} & "
            f"{row['b_clean']} & {row['b_key']} & {row['b_tpl']} & {row['b_both']} & "
            f"{row['a_clean']} & {row['a_key']} & {row['a_tpl']} & {row['a_both']} & "
            f"{row['asr_m']} & {row['asr']} & {row['asr_M']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Attack success rates (ASR) by model and topology. ASR-m/ASR/ASR-M denote min/mean/max of ASR(Both) across topologies for a given model and optimization level.}")
    lines.append(r"\label{tab:main_asr}")
    lines.append(r"\end{table*}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    # Experiment parameters
    N_SEG = 50
    N_BENIGN = 19
    EPISODES = 10

    topologies = ["star", "chain", "dag"]
    opt_levels = ["none", "routing", "routing_key", "full"]

    account_positions = build_account_positions(N_SEG)

    base_cfg = {
        "n_segments": N_SEG,
        "n_benign": N_BENIGN,
        "account_affinity": 0.20,
        "route_bias": 0.0,
        "key_index": 0,
        "template_slot": "prefix",
        "tampered_agent": "tempered_agent",
        "max_new_tokens": 64,
    }

    out_dir = "outputs"
    ensure_dir(out_dir)

    csv_rows = []
    table_rows = []

    for model_label, model_name in MODELS.items():
        print("\n===========================================")
        print(f"MODEL = {model_label}")
        print("===========================================")

        for level in opt_levels:
            theta = train_surrogate(
                n_segments=N_SEG,
                account_positions=account_positions,
                level=level,
                steps=300 if level != "none" else 0,
            )

            route_bias_star = theta["route_bias"] if level != "none" else 0.0
            key_star = theta["key_index"] if level in ("routing_key", "full") else 0
            tpl_star = theta["template_slot"] if level == "full" else "prefix"

            after_both_across = []

            for topo in topologies:
                print(f"\nTOPO={topo}  OPT={level}")

                before_cfg = dict(base_cfg)
                before_cfg["topology"] = topo
                before_cfg["opt_level"] = level + "_before"
                print("\n[BEFORE]")
                before = evaluate(before_cfg, EPISODES, model_name, n_segments=N_SEG)
                print(f"[BEFORE ASR] clean={before['clean']:.3f} key={before['key_only']:.3f} "
                      f"tmpl={before['template_only']:.3f} both={before['both']:.3f}")


                after_cfg = dict(base_cfg)
                after_cfg["topology"] = topo
                after_cfg["opt_level"] = level + "_after"
                after_cfg["route_bias"] = float(route_bias_star)
                after_cfg["key_index"] = int(key_star)
                after_cfg["template_slot"] = str(tpl_star)

                print("\n[AFTER]")
                after = evaluate(after_cfg, EPISODES, model_name, n_segments=N_SEG)
                print(f"[AFTER  ASR] clean={after['clean']:.3f} key={after['key_only']:.3f} "
                      f"tmpl={after['template_only']:.3f} both={after['both']:.3f}")


                after_both_across.append(after["both"])

                rec = {
                    "model": model_label,
                    "model_name": model_name,
                    "topology": topo,
                    "level": level,
                    "theta_route_bias": after_cfg["route_bias"],
                    "theta_key_index": after_cfg["key_index"],
                    "theta_template_slot": after_cfg["template_slot"],
                    "before_clean": before["clean"],
                    "before_key_only": before["key_only"],
                    "before_template_only": before["template_only"],
                    "before_both": before["both"],
                    "after_clean": after["clean"],
                    "after_key_only": after["key_only"],
                    "after_template_only": after["template_only"],
                    "after_both": after["both"],
                }
                csv_rows.append(rec)

            mn, mean, mx = agg_min_mean_max(after_both_across)

            for topo in topologies:
                rr = [r for r in csv_rows if r["model"] == model_label and r["level"] == level and r["topology"] == topo][-1]
                table_rows.append({
                    "model": model_label,
                    "topology": topo,
                    "level": level,
                    "b_clean": format_pct(rr["before_clean"]),
                    "b_key": format_pct(rr["before_key_only"]),
                    "b_tpl": format_pct(rr["before_template_only"]),
                    "b_both": format_pct(rr["before_both"]),
                    "a_clean": format_pct(rr["after_clean"]),
                    "a_key": format_pct(rr["after_key_only"]),
                    "a_tpl": format_pct(rr["after_template_only"]),
                    "a_both": format_pct(rr["after_both"]),
                    "asr_m": format_pct(mn),
                    "asr": format_pct(mean),
                    "asr_M": format_pct(mx),
                })

            print("\n[AGG] model:", model_label, "level:", level)
            print("ASR-m/min:", mn, "ASR/mean:", mean, "ASR-M/max:", mx)
            print("theta*:", theta)

    csv_path = os.path.join(out_dir, "results.csv")
    write_csv(
        csv_path,
        csv_rows,
        fieldnames=list(csv_rows[0].keys()) if csv_rows else [],
    )

    tex_path = os.path.join(out_dir, "table1.tex")
    write_table1_tex(tex_path, table_rows)

    print("\n===========================================")
    print("DONE. Wrote:")
    print(" -", csv_path)
    print(" -", tex_path)
    print("===========================================")

if __name__ == "__main__":
    main()
