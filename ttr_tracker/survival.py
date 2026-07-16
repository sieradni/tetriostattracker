from math import sqrt


def kaplan_meier(times: list[float], events: list[bool]) -> list[dict]:
    """Kaplan-Meier survival estimate.

    times:  list of observed times (pieces, seconds, score, etc.)
    events: True = event occurred (misdrop), False = censored (completed run)

    Returns list of dicts:
        {"time": t, "survival": S(t), "std_err": se, "at_risk": n, "events": d}
    """
    paired = sorted(zip(times, events), key=lambda x: x[0])
    n = len(paired)
    if n == 0:
        return []

    result: list[dict] = []
    survival = 1.0
    var_sum = 0.0
    at_risk = n
    i = 0

    while i < n:
        t = paired[i][0]
        d = 0
        c = 0
        while i < n and paired[i][0] == t:
            if paired[i][1]:
                d += 1
            else:
                c += 1
            i += 1

        if d > 0:
            survival *= (at_risk - d) / at_risk
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
        std_err = sqrt(var_sum) * survival if var_sum > 0 else 0.0

        result.append({
            "time": t,
            "survival": round(survival, 4),
            "std_err": round(std_err, 4),
            "at_risk": at_risk,
            "events": d,
        })

        at_risk -= d + c

    return result


def survival_percentiles(curve: list[dict], pcts: list[float]) -> list[dict]:
    """Extract time values at given survival percentiles.

    curve: output of kaplan_meier()
    pcts:  list of percentiles e.g. [0.25, 0.50, 0.75]

    Returns list of {"percentile": p, "time": t_or_none}
    """
    result = []
    for pct in pcts:
        t = None
        for pt in curve:
            if pt["survival"] <= pct:
                t = pt["time"]
                break
        result.append({"percentile": pct, "time": t})
    return result


def survival_at_thresholds(curve: list[dict], thresholds: list[float]) -> list[dict]:
    """Get survival probability at specific time thresholds.

    curve:      output of kaplan_meier()
    thresholds: list of time values to evaluate at

    Returns list of {"threshold": t, "survival": S(t), "std_err": se}
    """
    result = []
    for thr in sorted(thresholds):
        surv = 1.0
        se = 0.0
        for pt in curve:
            if pt["time"] > thr:
                break
            surv = pt["survival"]
            se = pt["std_err"]
        result.append({"threshold": thr, "survival": surv, "std_err": se})
    return result


def compute_likelihoods(
    full_runs: list[dict],
    partial_runs: list[dict],
) -> dict:
    """Compute survival curves and likelihood summaries.

    full_runs:    list of dicts with 'pieces', 'time', 'score'
    partial_runs: list of dicts with 'pieces', 'time', 'score'

    Returns dict with keys:
        pieces_curve, time_curve, score_curve,
        pieces_percentiles, time_percentiles, score_percentiles,
        pieces_thresholds, time_thresholds, score_thresholds
    """
    pieces_times = [r["pieces"] for r in full_runs]
    pieces_events = [False] * len(full_runs)
    pieces_times += [r["pieces"] for r in partial_runs]
    pieces_events += [True] * len(partial_runs)
    pieces_curve = kaplan_meier(pieces_times, pieces_events)

    time_times = [r["time"] for r in full_runs]
    time_events = [False] * len(full_runs)
    time_times += [r["time"] for r in partial_runs]
    time_events += [True] * len(partial_runs)
    time_curve = kaplan_meier(time_times, time_events)

    score_times = [r["score"] for r in full_runs]
    score_events = [False] * len(full_runs)
    score_times += [r["score"] for r in partial_runs]
    score_events += [True] * len(partial_runs)
    score_curve = kaplan_meier(score_times, score_events)

    percentiles = [0.25, 0.50, 0.75, 0.90]

    return {
        "pieces_curve": pieces_curve,
        "time_curve": time_curve,
        "score_curve": score_curve,
        "pieces_percentiles": survival_percentiles(pieces_curve, percentiles),
        "time_percentiles": survival_percentiles(time_curve, percentiles),
        "score_percentiles": survival_percentiles(score_curve, percentiles),
    }
