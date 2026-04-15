import random

def star(segments: list[str]) -> list[tuple[str, int]]:
    return [(seg, 0) for seg in segments]

def chain(segments: list[str], hops_per_segment: int = 3) -> list[tuple[str, int]]:
    out = []
    for seg in segments:
        for h in range(hops_per_segment):
            out.append((seg, h))
    return out

def dag(segments: list[str], extra_branch_prob: float = 0.5) -> list[tuple[str, int]]:
    out = []
    for seg in segments:
        out.append((seg, 0))
        if random.random() < extra_branch_prob:
            out.append((seg, 1))
    return out

TOPOLOGIES = {
    "star": lambda segs: star(segs),
    "chain": lambda segs: chain(segs, hops_per_segment=3),
    "dag": lambda segs: dag(segs, extra_branch_prob=0.5),
}

