#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecast.

Implements /srv/projects/gta6-trophy-rarity-forecast-01/forecast-brief.md
with the corrections in forecast-notes.md, which replace the matching
lines of the brief. Do not add a second model. Do not fit the practice
answers.

Idea, in one line: an earn rate is holders / tracked players, so fit
those two series separately (log holders and log players against
ln(1 + hours since launch)), shrink both slopes by the same factor
0.75, divide the two forecasts to get the median rate, and put a
narrow asymmetric band around it on the logit scale.

Corrections from forecast-notes.md, in short:
- One shrink factor, 0.75, for both slopes. A negative slope means
  "no growth": set the target level to the last observation and do
  not use the intercept.
- Fix from forecast-fix.md: project from the last observation, not
  from the intercept. N_target = N_last * exp(0.75 * b_N *
  (x_target - x_last)), and likewise E_target = E_last * exp(0.75 *
  b_E * (x_target - x_last_E)). The intercept is used only to fit
  the slope, never to set the level. A positive slope must therefore
  move the forecast above the last observation when the target is
  after the last read.
- Guard the regression: fewer than 3 points, or x values not all
  distinct within 1e-9, or a zero denominator, means no fit. Then
  use the last positive N and the last positive E for that trophy.
- A null rate is skipped for that trophy only, not for the whole
  read. The last numeric rate is the last non-null rate.
- Round every quantile to 6 decimal places, then make them
  non-decreasing again.

Python 3.13, standard library only. No network, no randomness,
no clock read, no files read except the case files. Two runs on the
same input write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Normal quantiles for 5%, 25%, 50%, 75%, 95%.
Z_VALUES = [-1.64485362695, -0.674489750196, 0.0,
            0.674489750196, 1.64485362695]

SHRINK = 0.75


def parse_time(value):
    """Parse an ISO 8601 UTC timestamp, or return None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value))


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit(p):
    """logit of a percent, after clamping to [0.01, 99.99]."""
    p = clamp(p, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    """100 / (1 + exp(-z)), clamped to [0, 100], overflow-safe."""
    if z >= 0:
        return clamp(100.0 / (1.0 + math.exp(-z)), 0.0, 100.0)
    exp_z = math.exp(z)
    return clamp(100.0 * exp_z / (1.0 + exp_z), 0.0, 100.0)


def hours_since_origin(moment):
    return (moment - ORIGIN).total_seconds() / 3600.0


def log_time(h):
    """ln(1 + h), guarded so a slightly negative h cannot crash us."""
    return math.log(1.0 + h if 1.0 + h > 1e-12 else 1e-12)


def fit_log_series(xs, ys):
    """OLS y = a + b*x, solved by hand, no numpy.

    Returns (a, b), or None when there is no fit: fewer than 3
    points, x values not all distinct within 1e-9, or a zero
    denominator. A zero denominator means "no fit", not a crash
    and not a slope of 0 through the intercept.
    """
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    ordered = sorted(xs)
    for k in range(1, len(ordered)):
        if ordered[k] - ordered[k - 1] < 1e-9:
            return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var
    return mean_y - slope * mean_x, slope


def quantiles(median, sigma_down, sigma_up):
    """Five quantiles of the rate at target, from median + logit band.

    Clamp to [0, 100], make non-decreasing, round to 6 decimal
    places, then make non-decreasing again (notes, section 3).
    """
    centre = logit(median)
    out = []
    for z in Z_VALUES:
        if z < 0:
            sigma = sigma_down
        elif z > 0:
            sigma = sigma_up
        else:
            sigma = sigma_down
        out.append(inv_logit(centre + z * sigma))
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    out = [clamp(round(v, 6), 0.0, 100.0) + 0.0 for v in out]
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return out


def forecast_case(case):
    """Forecast one case dict, returning {"trophies": [...]}."""
    trophies = case.get("trophies") if isinstance(case, dict) else None
    if not isinstance(trophies, list):
        trophies = []
    ids = []
    for trophy in trophies:
        if isinstance(trophy, dict) and is_number(trophy.get("id")):
            ids.append(int(trophy["id"]))
    n_trophies = len(ids)

    target = parse_time(case.get("target")) if isinstance(case, dict) else None
    if target is None and isinstance(case, dict):
        # Fall back to at + horizon_minutes, only if target is missing.
        at = parse_time(case.get("at"))
        horizon = case.get("horizon_minutes")
        if at is not None and is_number(horizon):
            target = datetime.fromtimestamp(
                at.timestamp() + float(horizon) * 60.0, tz=timezone.utc)
    if target is None:
        return {"trophies": [{"id": tid, "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                             for tid in sorted(ids)]}

    h_target = hours_since_origin(target)
    x_target = log_time(h_target)

    history = case.get("history") if isinstance(case, dict) else None
    if not isinstance(history, list):
        history = []

    # --- Usable reads: a parseable time and positive players.
    # A null rate is skipped for that trophy only (notes, section 2),
    # so a read is NOT discarded because one trophy's rate is null.
    # rates[i] is a float, or None when that trophy has no rate here.
    usable = []
    for read in history:
        if not isinstance(read, dict):
            continue
        moment = parse_time(read.get("at"))
        if moment is None:
            continue
        players = read.get("players_tracked")
        if not is_number(players) or players <= 0:
            continue
        raw_rates = read.get("rates")
        if not isinstance(raw_rates, list):
            raw_rates = []
        rates = []
        for i in range(n_trophies):
            if i < len(raw_rates) and is_number(raw_rates[i]):
                rates.append(float(raw_rates[i]))
            else:
                rates.append(None)
        usable.append({
            "h": hours_since_origin(moment),
            "x": log_time(hours_since_origin(moment)),
            "n": float(players),
            "rates": rates,
        })

    # --- First refresh: first read whose N is >= 2x the first read's N.
    # Fit only reads at or after it; if none doubles, fit all usable. ---
    if usable:
        first_n = usable[0]["n"]
        first_refresh = None
        for j, read in enumerate(usable):
            if read["n"] >= 2.0 * first_n:
                first_refresh = j
                break
        fitted = usable[first_refresh:] if first_refresh is not None else usable
    else:
        fitted = []

    def last_rate(index):
        """Last numeric (non-null) rate for a trophy, in history order."""
        for read in reversed(history):
            if isinstance(read, dict):
                rates = read.get("rates")
                if (isinstance(rates, list) and len(rates) > index
                        and is_number(rates[index])):
                    return float(rates[index])
        return None

    def last_positive_holders(index):
        """Last positive holders E for a trophy, among fitted reads."""
        last_e = None
        for read in fitted:
            rate = read["rates"][index]
            if rate is None:
                continue
            holders = rate / 100.0 * read["n"]
            if holders > 0 and math.isfinite(holders):
                last_e = holders
        return last_e

    rows = []

    # --- No-regression fallback: fewer than 3 fitted reads.
    # Use the last positive N and the last positive E for that trophy.
    if len(fitted) < 3:
        last_n = fitted[-1]["n"] if fitted else None
        for i, tid in enumerate(ids):
            last_e = last_positive_holders(i)
            if last_n is not None and last_e is not None:
                median = 100.0 * last_e / last_n
            else:
                last = last_rate(i)
                median = last if last is not None else 0.0
            rows.append({"id": tid,
                         "q": quantiles(median, 0.35, 0.35)})
        rows.sort(key=lambda row: row["id"])
        return {"trophies": rows}

    # --- Players at the target: OLS of ln(N) on ln(1+h), slope x0.75.
    # Project from the last observation, not from the intercept
    # (forecast-fix.md): N_target = N_last * exp(0.75 * b_N *
    # (x_target - x_last)). Negative slope or no fit: N_target is
    # the last N.
    last_n = fitted[-1]["n"]
    x_last_n = fitted[-1]["x"]
    players_fit = fit_log_series(
        [read["x"] for read in fitted],
        [math.log(read["n"]) for read in fitted])
    if players_fit is None:
        n_target = last_n
    else:
        _a_n, b_n = players_fit
        if b_n <= 0:
            n_target = last_n
        else:
            try:
                n_target = last_n * math.exp(
                    SHRINK * b_n * (x_target - x_last_n))
            except OverflowError:
                n_target = last_n
            if not math.isfinite(n_target):
                n_target = last_n
            n_target = max(n_target, last_n)

    # --- Holders at the target, trophy by trophy. ---
    for i, tid in enumerate(ids):
        points = []          # (x, ln(E)) for positive holders
        last_positive_e = None
        last_positive_x = None
        for read in fitted:
            rate = read["rates"][i]
            if rate is None:
                continue
            holders = rate / 100.0 * read["n"]
            if holders > 0 and math.isfinite(holders):
                points.append((read["x"], math.log(holders)))
                last_positive_e = holders
                last_positive_x = read["x"]

        if len(points) < 3 or last_positive_e is None:
            # No holders regression for this trophy: median from the
            # last positive E over the target players, band 0.35.
            if last_positive_e is not None:
                median = 100.0 * last_positive_e / n_target
            else:
                last = last_rate(i)
                median = last if last is not None else 0.0
            rows.append({"id": tid,
                         "q": quantiles(median, 0.35, 0.35)})
            continue

        holders_fit = fit_log_series(
            [p[0] for p in points], [p[1] for p in points])
        if holders_fit is None:
            # Guarded out: use the last positive E, band 0.35.
            median = 100.0 * last_positive_e / n_target
            rows.append({"id": tid,
                         "q": quantiles(median, 0.35, 0.35)})
            continue

        _a_i, b_i = holders_fit
        if b_i <= 0:
            # Negative slope: no growth, E_target is the last E,
            # no intercept. This is still a fitted trophy, so the
            # band is the normal rare/common band, not 0.35.
            e_target = last_positive_e
        else:
            # Project from the last positive observation, not from
            # the intercept (forecast-fix.md): E_target = E_last *
            # exp(0.75 * b_E * (x_target - x_last_E)).
            try:
                e_target = last_positive_e * math.exp(
                    SHRINK * b_i * (x_target - last_positive_x))
            except OverflowError:
                e_target = last_positive_e
            if not math.isfinite(e_target):
                e_target = last_positive_e
            # Holders never shrink below the last seen, never exceed players.
            e_target = max(e_target, last_positive_e)
            e_target = min(e_target, n_target)
        median = 100.0 * e_target / n_target

        last = last_rate(i)
        if last is not None and last < 10:
            sigma_down, sigma_up = 0.45, 0.25
        else:
            sigma_down, sigma_up = 0.30, 0.30
        rows.append({"id": tid,
                     "q": quantiles(median, sigma_down, sigma_up)})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(input_dir):
        return 0
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        in_path = os.path.join(input_dir, name)
        out_name = name[:-len(".in")] + ".out"
        out_path = os.path.join(output_dir, out_name)
        try:
            with open(in_path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
            text = json.dumps(result, ensure_ascii=False,
                              separators=(",", ":"))
        except Exception:
            # One bad case must not crash the whole run.
            continue
        try:
            with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except Exception:
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
