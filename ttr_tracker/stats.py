BLITZ_DURATION = 120.0


def compute_time_elapsed(time_left_seconds: float) -> float:
    time_left = max(0.0, min(time_left_seconds, BLITZ_DURATION))
    return BLITZ_DURATION - time_left


def compute_kps(inputs: int, elapsed_seconds: float) -> float:
    return inputs / elapsed_seconds if elapsed_seconds > 0 else 0.0


def compute_kpp(inputs: int, pieces: int) -> float:
    return inputs / pieces if pieces else 0.0
