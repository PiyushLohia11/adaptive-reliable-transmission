"""Adaptive transmission controller for a lossy-network simulator.

The controller is deliberately dependency-free and deterministic. It consumes
interval telemetry and returns bounded congestion-window, pacing, and optional
redundancy adjustments. It is a rule-based adaptive baseline, not a trained RL
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class NetworkMetrics:
    """Measurements for one observation interval.

    Attributes:
        packet_loss_rate: Lost packets / transmitted packets, from 0.0 to 1.0.
        delay_ms: Measured one-way or round-trip delay in milliseconds. Pick one
            definition and use it consistently throughout an experiment.
        baseline_delay_ms: Minimum or unloaded delay for this same path.
        congestion: Normalized congestion indicator from 0.0 (none) to 1.0
            (severe), such as queue occupancy / queue capacity.
    """

    packet_loss_rate: float
    delay_ms: float
    baseline_delay_ms: float
    congestion: float


@dataclass(frozen=True)
class TransmissionAction:
    """Control outputs; multipliers are applied to the current sender settings."""

    cwnd_multiplier: float
    pacing_multiplier: float
    redundancy_fraction: float
    congestion_pressure: float
    reason: str


class AdaptiveController:
    """Smoothed bounded controller that reacts differently to congestion loss.

    Call :meth:`update` once per fixed observation interval. The output's
    ``cwnd_multiplier`` and ``pacing_multiplier`` are multipliers for the
    current congestion window and pacing rate. ``redundancy_fraction`` is an
    optional FEC/parity fraction for a simulator that supports it; it does not
    replace retransmission-based reliability.
    """

    def __init__(self, smoothing: float = 0.35) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self._loss: Optional[float] = None
        self._delay_inflation: Optional[float] = None
        self._congestion: Optional[float] = None

    @staticmethod
    def _bounded(name: str, value: float, low: float, high: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        return value

    def update(self, metrics: NetworkMetrics) -> TransmissionAction:
        loss = self._bounded("packet_loss_rate", metrics.packet_loss_rate, 0.0, 1.0)
        delay = self._bounded("delay_ms", metrics.delay_ms, 0.0, float("inf"))
        base = self._bounded("baseline_delay_ms", metrics.baseline_delay_ms, 1e-9, float("inf"))
        queue = self._bounded("congestion", metrics.congestion, 0.0, 1.0)

        # Delay inflation estimates queuing independently of propagation delay.
        delay_pressure = min(1.0, max(0.0, delay / base - 1.0) / 2.0)
        if self._loss is None:
            self._loss, self._delay_inflation, self._congestion = loss, delay_pressure, queue
        else:
            a = self.smoothing
            self._loss = a * loss + (1.0 - a) * self._loss
            self._delay_inflation = a * delay_pressure + (1.0 - a) * self._delay_inflation  # type: ignore[operator]
            self._congestion = a * queue + (1.0 - a) * self._congestion  # type: ignore[operator]

        smoothed_loss = self._loss
        smoothed_delay = self._delay_inflation
        smoothed_queue = self._congestion
        # Weighted pressure combines all three required telemetry inputs.
        pressure = min(1.0, 0.40 * smoothed_queue + 0.35 * smoothed_delay + 0.25 * smoothed_loss)

        # When delay/queue are low, loss is more likely to be wireless/random;
        # preserve most sending capacity and use bounded optional redundancy.
        noncongestion_loss = smoothed_loss * (1.0 - max(smoothed_queue, smoothed_delay))
        cwnd_multiplier = min(1.08, max(0.70, 1.04 - 0.42 * pressure - 0.08 * noncongestion_loss))
        pacing_multiplier = min(1.06, max(0.65, 1.03 - 0.58 * pressure - 0.10 * smoothed_queue))
        redundancy = min(0.25, max(0.0, 0.50 * noncongestion_loss))

        if pressure >= 0.55:
            reason = "high congestion: reduce sending rate"
        elif pressure >= 0.25:
            reason = "moderate congestion: cautiously reduce sending rate"
        elif noncongestion_loss >= 0.03:
            reason = "loss without strong queue evidence: preserve rate and add bounded redundancy"
        else:
            reason = "low network stress: cautiously probe available capacity"

        return TransmissionAction(
            cwnd_multiplier=round(cwnd_multiplier, 4),
            pacing_multiplier=round(pacing_multiplier, 4),
            redundancy_fraction=round(redundancy, 4),
            congestion_pressure=round(pressure, 4),
            reason=reason,
        )

    def reset(self) -> None:
        """Clear telemetry history between independent simulation episodes."""
        self._loss = self._delay_inflation = self._congestion = None


def apply_action(current_cwnd_packets: float, current_pacing_pps: float,
                 action: TransmissionAction) -> tuple[float, float]:
    """Return updated (cwnd_packets, pacing_pps), each bounded to >= 1."""
    if current_cwnd_packets <= 0 or current_pacing_pps <= 0:
        raise ValueError("current cwnd and pacing rate must be positive")
    return (
        max(1.0, current_cwnd_packets * action.cwnd_multiplier),
        max(1.0, current_pacing_pps * action.pacing_multiplier),
    )


if __name__ == "__main__":
    scenarios = {
        "clean": NetworkMetrics(0.0, 50.0, 50.0, 0.05),
        "random_loss_low_queue": NetworkMetrics(0.10, 55.0, 50.0, 0.05),
        "moderate_congestion": NetworkMetrics(0.04, 100.0, 50.0, 0.45),
        "severe_congestion": NetworkMetrics(0.12, 180.0, 50.0, 0.90),
    }
    for name, sample in scenarios.items():
        controller = AdaptiveController()
        action = controller.update(sample)
        cwnd, pacing = apply_action(64, 1000, action)
        print(f"{name:24} {action} -> cwnd={cwnd:.1f} packets, pacing={pacing:.1f} pps")
