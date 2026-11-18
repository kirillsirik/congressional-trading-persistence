"""Deterministic long-after-purchase lot engine (POR-001..POR-013).

Minimal v2 implementation of the contract portfolio keys: strict-next-close
entry, 20-trading-day separate lots, joint pro-rata routing of same-date
signals (input order can never decide allocation), aggregate security /
household / gross caps, daily risk-free cash accrual, one-way costs charged
once per executed side, exposure-matched benchmark, and reason-coded
missing-price dispositions (never zero-filled). The terminal rule is a
PENDING HS-06 parameter: real-data runs refuse until signed; synthetic runs
must pass an explicit SYNTHETIC_RULE_* label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.timing import FOLLOWER_ENTRY_COL


class PortfolioContractError(RuntimeError):
    pass


#: the locked design's terminal rule (final directive §18): all remaining
#: positions are liquidated at the sample endpoint.
TERMINAL_LIQUIDATE_ALL = "liquidate_all_at_sample_end"


@dataclass(frozen=True)
class PortfolioParams:
    holding_days: int = 20
    signal_target_fraction: float = 0.05
    security_cap: float = 0.05
    household_cap: float = 0.10
    gross_cap: float = 1.00
    one_way_cost_bp: float = 10.0
    max_open_lots: int | None = None
    terminal_rule: str | None = None
    data_is_real: bool = False

    def validate(self) -> None:
        if self.data_is_real and self.terminal_rule != TERMINAL_LIQUIDATE_ALL:
            raise PortfolioContractError(
                "real-data simulation requires the locked terminal rule "
                f"{TERMINAL_LIQUIDATE_ALL!r} (final directive §18); got "
                f"{self.terminal_rule!r}"
            )
        if self.terminal_rule is None:
            raise PortfolioContractError(
                "runs must pass an explicit terminal rule "
                f"({TERMINAL_LIQUIDATE_ALL!r} or a SYNTHETIC_RULE_* label)"
            )


@dataclass
class _Lot:
    lot_id: str
    signal_uid: str
    permno: int
    filer_uid: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    value: float
    entry_notional: float
    disposition: str = "open"


@dataclass
class PortfolioResult:
    daily: pd.DataFrame
    lots: pd.DataFrame
    routing: pd.DataFrame
    params: PortfolioParams = field(default=None)  # type: ignore[assignment]
    initial_equity: float = float("nan")


def _prorate(desired: dict[str, float], headroom: float) -> dict[str, float]:
    total = sum(desired.values())
    if total <= headroom or total <= 0:
        return desired
    scale = headroom / total
    return {k: v * scale for k, v in desired.items()}


def simulate_long_after_purchase(
    purchase_signals: pd.DataFrame,
    stock_returns: dict[int, pd.Series],
    benchmark_returns: pd.Series,
    risk_free: pd.Series,
    calendar: pd.DatetimeIndex,
    params: PortfolioParams,
    initial_equity: float = 1_000_000.0,
) -> PortfolioResult:
    params.validate()
    required = {"signal_uid", "permno", "filer_uid", FOLLOWER_ENTRY_COL}
    missing = required - set(purchase_signals.columns)
    if missing:
        raise KeyError(f"purchase signals missing {sorted(missing)}")

    # Canonical processing order: input row order may not decide allocation
    # (POR-002); everything below is group arithmetic over a sorted frame.
    signals = purchase_signals.sort_values(
        ["signal_uid"], kind="mergesort"
    ).reset_index(drop=True)

    cost = params.one_way_cost_bp / 1e4
    cash = float(initial_equity)
    bench_equity = float(initial_equity)
    lots: list[_Lot] = []
    closed: list[_Lot] = []
    routing_rows: list[dict] = []
    daily_rows: list[dict] = []
    by_entry = dict(tuple(signals.groupby(FOLLOWER_ENTRY_COL)))
    prev_risky_frac = 0.0

    for day in calendar:
        rf_day = float(risk_free.get(day, np.nan))
        if np.isnan(rf_day):
            raise PortfolioContractError(
                f"missing risk-free rate on {day.date()}; zero fill is forbidden"
            )
        interest = cash * rf_day
        cash += interest
        costs_paid_today = 0.0

        # mark open lots with day's return; missing return is a disposition,
        # never a zero (POR-008)
        for lot in lots:
            r = stock_returns.get(lot.permno, pd.Series(dtype=float)).get(day, np.nan)
            if np.isnan(r):
                lot.disposition = "missing_return_day"
            else:
                lot.value *= 1 + float(r)
                if lot.disposition == "missing_return_day":
                    lot.disposition = "open"

        # exits at planned date's close, extended past missing-price days
        still_open = []
        for lot in lots:
            if day >= lot.exit_date and lot.disposition != "missing_return_day":
                proceeds = lot.value * (1 - cost)
                costs_paid_today += lot.value * cost
                cash += proceeds
                lot.disposition = (
                    "exited_planned" if day == lot.exit_date else "exited_deferred_missing_price"
                )
                closed.append(lot)
            else:
                still_open.append(lot)
        lots = still_open

        positions_value = sum(lot.value for lot in lots)
        equity = cash + positions_value

        # entries at this day's close
        todays = by_entry.get(day)
        if todays is not None and len(todays):
            desired = {
                row.signal_uid: equity * params.signal_target_fraction
                for row in todays.itertuples(index=False)
            }
            # planned exit must be observable under the terminal rule; with
            # terminal liquidation, late entries are held to the endpoint
            if params.terminal_rule != TERMINAL_LIQUIDATE_ALL:
                for row in todays.itertuples(index=False):
                    pos = calendar.searchsorted(day)
                    if pos + params.holding_days >= len(calendar):
                        routing_rows.append(
                            {
                                "signal_uid": row.signal_uid,
                                "state": "terminal_horizon_unavailable",
                                "executed_notional": 0.0,
                            }
                        )
                        desired.pop(row.signal_uid, None)
            sig_lookup = {r.signal_uid: r for r in todays.itertuples(index=False)}
            # maximum open-lot count (§18): reject in canonical sorted order
            if params.max_open_lots is not None:
                slots = max(0, params.max_open_lots - len(lots))
                for signal_uid in sorted(desired)[slots:]:
                    routing_rows.append(
                        {
                            "signal_uid": signal_uid,
                            "state": "max_open_lots_constraint",
                            "executed_notional": 0.0,
                        }
                    )
                    desired.pop(signal_uid)
            # security caps over aggregate active exposure (POR-005)
            for permno in {sig_lookup[s].permno for s in desired}:
                members = {
                    s: v for s, v in desired.items() if sig_lookup[s].permno == permno
                }
                active = sum(x.value for x in lots if x.permno == permno)
                headroom = max(0.0, params.security_cap * equity - active)
                desired.update(_prorate(members, headroom))
            # household caps
            for filer in {sig_lookup[s].filer_uid for s in desired}:
                members = {
                    s: v for s, v in desired.items() if sig_lookup[s].filer_uid == filer
                }
                active = sum(x.value for x in lots if x.filer_uid == filer)
                headroom = max(0.0, params.household_cap * equity - active)
                desired.update(_prorate(members, headroom))
            # gross exposure and cash, jointly (POR-003); entry cost paid from cash
            gross_headroom = max(0.0, params.gross_cap * equity - positions_value)
            cash_headroom = max(0.0, cash / (1 + cost))
            desired = _prorate(desired, min(gross_headroom, cash_headroom))

            for signal_uid, notional in sorted(desired.items()):
                row = sig_lookup[signal_uid]
                if notional <= 1e-9:
                    routing_rows.append(
                        {
                            "signal_uid": signal_uid,
                            "state": "cash_or_cap_constraint",
                            "executed_notional": 0.0,
                        }
                    )
                    continue
                pos = calendar.searchsorted(day)
                exit_pos = min(pos + params.holding_days, len(calendar) - 1)
                exit_date = calendar[exit_pos]
                cash -= notional * (1 + cost)
                costs_paid_today += notional * cost
                lots.append(
                    _Lot(
                        lot_id=f"lot-{signal_uid}",
                        signal_uid=signal_uid,
                        permno=int(row.permno),
                        filer_uid=str(row.filer_uid),
                        entry_date=day,
                        exit_date=exit_date,
                        value=notional,
                        entry_notional=notional,
                    )
                )
                routing_rows.append(
                    {
                        "signal_uid": signal_uid,
                        "state": "accepted",
                        "executed_notional": notional,
                    }
                )
            positions_value = sum(lot.value for lot in lots)
            equity = cash + positions_value

        if abs(equity - (cash + positions_value)) > 1e-6:
            raise PortfolioContractError(f"accounting identity failed on {day.date()}")

        # exposure-matched benchmark (POR-011): same risky/cash split as held
        # entering the day
        bench_ret = float(benchmark_returns.get(day, np.nan))
        if np.isnan(bench_ret):
            raise PortfolioContractError(f"missing benchmark return on {day.date()}")
        bench_equity *= 1 + prev_risky_frac * bench_ret + (1 - prev_risky_frac) * rf_day
        prev_risky_frac = positions_value / equity if equity > 0 else 0.0

        by_security: dict[int, float] = {}
        for lot in lots:
            by_security[lot.permno] = by_security.get(lot.permno, 0.0) + lot.value
        max_sec_frac = max(by_security.values()) / equity if by_security and equity > 0 else 0.0
        daily_rows.append(
            {
                "date": day,
                "cash": cash,
                "positions_value": positions_value,
                "equity": equity,
                "benchmark_equity": bench_equity,
                "risky_fraction": prev_risky_frac,
                "interest": interest,
                "costs_paid": costs_paid_today,
                "n_open_lots": len(lots),
                "max_security_exposure_frac": max_sec_frac,
            }
        )

    # terminal treatment for still-open lots under the declared rule
    if params.terminal_rule == TERMINAL_LIQUIDATE_ALL and lots:
        cost = params.one_way_cost_bp / 1e4
        terminal_costs = 0.0
        for lot in lots:
            proceeds = lot.value * (1 - cost)
            terminal_costs += lot.value * cost
            cash += proceeds
            lot.disposition = "liquidated_at_sample_end"
            closed.append(lot)
        lots = []
        last = daily_rows[-1]
        last["cash"] = cash
        last["positions_value"] = 0.0
        last["equity"] = cash
        last["costs_paid"] += terminal_costs
        last["n_open_lots"] = 0
        last["risky_fraction"] = 0.0
        last["max_security_exposure_frac"] = 0.0
    else:
        for lot in lots:
            lot.disposition = f"open_at_terminal::{params.terminal_rule}"
            closed.append(lot)

    daily = pd.DataFrame(daily_rows).set_index("date")
    # day-1 return is measured against the true initial account value, so the
    # initial->first-close interval is a realized return, never a NaN
    prev_equity = daily["equity"].shift(1)
    prev_equity.iloc[0] = initial_equity
    daily["strategy_return"] = daily["equity"] / prev_equity - 1
    prev_bench = daily["benchmark_equity"].shift(1)
    prev_bench.iloc[0] = initial_equity
    daily["benchmark_return"] = daily["benchmark_equity"] / prev_bench - 1
    daily.attrs["initial_equity"] = float(initial_equity)
    lots_frame = pd.DataFrame(
        [
            {
                "lot_id": lot.lot_id,
                "signal_uid": lot.signal_uid,
                "permno": lot.permno,
                "filer_uid": lot.filer_uid,
                "entry_date": lot.entry_date,
                "exit_date": lot.exit_date,
                "entry_notional": lot.entry_notional,
                "final_value": lot.value,
                "disposition": lot.disposition,
            }
            for lot in closed
        ]
    )
    routing = pd.DataFrame(routing_rows)
    # every signal has exactly one terminal routing state (POR-001)
    routed = set(routing["signal_uid"]) if len(routing) else set()
    unrouted = set(signals["signal_uid"]) - routed
    for signal_uid in sorted(unrouted):
        routing = pd.concat(
            [
                routing,
                pd.DataFrame(
                    [
                        {
                            "signal_uid": signal_uid,
                            "state": "timing_unavailable",
                            "executed_notional": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return PortfolioResult(
        daily=daily,
        lots=lots_frame,
        routing=routing,
        params=params,
        initial_equity=float(initial_equity),
    )
