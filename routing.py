import random
from typing import List

DUMMY_KEY = "KEYWORD_X"


def make_segments(n_segments: int) -> list[str]:
    segs = []
    for i in range(n_segments):
        if i % 10 == 0:
            segs.append(f"[ACCOUNT] segment_{i}: verify reservation details and user record lookup")
        elif i % 3 == 0:
            segs.append(f"[FLIGHT] segment_{i}: search flights and compare options")
        elif i % 3 == 1:
            segs.append(f"[HOTEL] segment_{i}: find hotels and shortlist")
        else:
            segs.append(f"[PLAN] segment_{i}: create itinerary and schedule")
    return segs


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def route_segment(
    segment: str,
    agents: List[str],
    key: str,
    route_bias: float,
    tempered_agent: str,
    account_affinity: float = 0.20,
) -> str:
    I_acc = 1.0 if ("[ACCOUNT]" in segment) else 0.0
    I_key = 1.0 if (key in segment) else 0.0

    rho = _clamp01(float(route_bias))
    a0 = _clamp01(float(account_affinity))

    p_tampered = _clamp01(a0 * I_acc + rho * I_acc * I_key)

    u = random.random()
    if u < p_tampered:
        return tempered_agent

    if not agents:
        return tempered_agent
    return random.choice(agents)
