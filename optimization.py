import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

TEMPLATE_SLOTS = ["prefix", "wrap", "suffix"]

def gumbel_softmax_sample(logits, tau=1.0):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    return torch.softmax((logits + g) / tau, dim=-1)

class SurrogateAttack(nn.Module):
    def __init__(self, n_segments: int, n_slots: int = 3):
        super().__init__()
        self.key_logits = nn.Parameter(torch.zeros(n_segments)) 
        self.slot_logits = nn.Parameter(torch.zeros(n_slots))    
        self.bias_logit = nn.Parameter(torch.zeros(1))          

    def forward(self, tau=1.0):
        key_prob = gumbel_softmax_sample(self.key_logits, tau=tau)    
        slot_prob = gumbel_softmax_sample(self.slot_logits, tau=tau)  
        bias = torch.sigmoid(self.bias_logit).squeeze().clamp(0, 1)
        return key_prob, slot_prob, bias

def surrogate_obj(key_prob, slot_prob, bias, account_positions):
    device = key_prob.device
    acc = account_positions.to(device).float()  

    eff = torch.tensor([1.0, 0.9, 0.6], device=device)  
    tmpl_eff = (slot_prob * eff).sum()

    p_key_on_acc = (key_prob * acc).sum()
    asr_both = p_key_on_acc * bias * tmpl_eff

    miss_key = (key_prob * (1.0 - acc)).sum()
    key_miss_penalty = 0.10 * miss_key

    far_key = 0.02 * bias
    far_tmpl = 0.01 * tmpl_eff
    utility_drop = 0.02 * bias

    key_entropy = -(key_prob * (key_prob + 1e-9).log()).sum()
    slot_entropy = -(slot_prob * (slot_prob + 1e-9).log()).sum()

    loss = (
        -asr_both
        + key_miss_penalty
        + far_key + far_tmpl
        + utility_drop
        - 0.001 * key_entropy
        - 0.001 * slot_entropy
    )
    return loss

@torch.no_grad()
def decode_theta(key_prob, slot_prob, bias):
    key_index = int(torch.argmax(key_prob).item())
    slot_index = int(torch.argmax(slot_prob).item())
    return {
        "route_bias": float(bias.item()),
        "key_index": key_index,
        "template_slot": TEMPLATE_SLOTS[slot_index],
    }

def train_surrogate(
    n_segments=50,
    account_positions=None,
    level="full",                 
    steps=300,
    lr=1e-2,
    tau_start=2.0,
    tau_end=0.5,
    seed=0,
):
    torch.manual_seed(seed)

    if account_positions is None:
        account_positions = torch.zeros(n_segments)
        for i in range(n_segments):
            if i % 10 == 0:
                account_positions[i] = 1.0

    model = SurrogateAttack(n_segments=n_segments, n_slots=len(TEMPLATE_SLOTS))

    if level == "routing":
        model.key_logits.requires_grad_(False)
        model.slot_logits.requires_grad_(False)
    elif level == "routing_key":
        model.slot_logits.requires_grad_(False)
    elif level == "full":
        pass
    elif level == "none":
        return {"route_bias": 0.0, "key_index": 0, "template_slot": "prefix"}
    else:
        raise ValueError(f"Unknown level={level}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)

    pbar = tqdm(range(steps), desc=f"opt({level})", leave=True)
    for i in pbar:
        frac = i / max(steps - 1, 1)
        tau = tau_start * (1 - frac) + tau_end * frac

        key_prob, slot_prob, bias = model(tau=tau)
        loss = surrogate_obj(key_prob, slot_prob, bias, account_positions)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 25 == 0 or i == steps - 1:
            theta = decode_theta(key_prob, slot_prob, bias)
            pbar.set_postfix({
                "loss": float(loss.item()),
                "bias": round(theta["route_bias"], 3),
                "key_idx": theta["key_index"],
                "slot": theta["template_slot"],
                "tau": round(tau, 3),
            })

    key_prob, slot_prob, bias = model(tau=tau_end)
    return decode_theta(key_prob, slot_prob, bias)
