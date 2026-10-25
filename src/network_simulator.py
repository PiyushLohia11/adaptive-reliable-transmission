"""Discrete-time single-bottleneck reliable packet-transfer simulator."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Literal

try:
    from .adaptive_controller import AdaptiveController, NetworkMetrics
    from .ppo_controller import Observation, PPOController
except ImportError:
    from adaptive_controller import AdaptiveController, NetworkMetrics
    from ppo_controller import Observation, PPOController


@dataclass(frozen=True)
class Scenario:
    name: str
    random_loss_rate: float
    base_rtt_ms: float
    capacity_pps: float
    queue_capacity_packets: int


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    mode: str
    seed: int
    throughput_mbps: float
    packet_delivery_ratio: float
    average_delay_ms: float
    retransmissions: int
    packets_sent: int
    elapsed_seconds: float
    completed: bool


SCENARIOS = (
    Scenario("clean_link", 0.00, 20.0, 1000.0, 100),
    Scenario("wireless_loss", 0.08, 20.0, 1000.0, 100),
    Scenario("high_delay", 0.02, 100.0, 1000.0, 100),
    Scenario("congested_bottleneck", 0.05, 40.0, 500.0, 24),
)


def simulate(
    scenario: Scenario,
    mode: Literal["reno", "heuristic", "ppo"],
    seed: int,
    total_packets: int = 5000,
    packet_size_bytes: int = 1200,
    max_epochs: int = 50000,
    ppo_agent: PPOController | None = None,
    training: bool = False,
) -> SimulationResult:
    """Transfer a finite workload through a lossy FIFO bottleneck.

    Each epoch is one baseline RTT. New and retransmitted packets share the
    same sender budget. Packets overflowed from the finite FIFO or lost on the
    wireless hop are retried in a later epoch. This is a teaching simulator,
    not an implementation of TCP, QUIC, or a calibrated physical channel.
    """
    if mode not in ("reno", "heuristic", "ppo"):
        raise ValueError("mode must be 'reno', 'heuristic', or 'ppo'")
    if mode == "ppo" and ppo_agent is None:
        raise ValueError("ppo mode requires a PPOController")
    if total_packets <= 0 or packet_size_bytes <= 0:
        raise ValueError("workload and packet size must be positive")
    if not 0.0 <= scenario.random_loss_rate <= 1.0:
        raise ValueError("random_loss_rate must be in [0, 1]")

    rng = random.Random(seed)
    dt = scenario.base_rtt_ms / 1000.0
    service_per_epoch = max(1, round(scenario.capacity_pps * dt))
    max_window = max(32, scenario.queue_capacity_packets * 2)
    cwnd = 16.0
    pacing_pps = cwnd / dt
    controller = AdaptiveController()
    agent = ppo_agent

    next_packet_id = 0
    retries: list[int] = []
    fifo: list[tuple[int, float]] = []
    delivered: set[int] = set()
    delivered_delays: list[float] = []
    transmission_attempts = 0
    retransmissions = 0
    total_loss_events = 0
    previous_loss_rate = 0.0
    time_s = 0.0
    previous_queue = 0

    for _epoch in range(max_epochs):
        if len(delivered) >= total_packets:
            break

        current_observation = Observation(
            loss_rate=previous_loss_rate,
            delay_ms=scenario.base_rtt_ms + (previous_queue / scenario.capacity_pps) * 1000.0,
            baseline_delay_ms=scenario.base_rtt_ms,
            congestion=min(1.0, previous_queue / scenario.queue_capacity_packets),
        )
        if mode == "heuristic":
            action = controller.update(NetworkMetrics(
                packet_loss_rate=previous_loss_rate,
                delay_ms=current_observation.delay_ms,
                baseline_delay_ms=scenario.base_rtt_ms,
                congestion=current_observation.congestion,
            ))
            cwnd = min(float(max_window), max(2.0, cwnd * action.cwnd_multiplier))
            pacing_pps = min(max_window / dt, max(1.0, pacing_pps * action.pacing_multiplier))
        elif mode == "ppo":
            assert agent is not None
            _, cwnd_multiplier, pacing_multiplier = agent.act(
                current_observation, deterministic=not training)
            cwnd = min(float(max_window), max(2.0, cwnd * cwnd_multiplier))
            pacing_pps = min(max_window / dt, max(1.0, pacing_pps * pacing_multiplier))
        else:
            # Reno-style AIMD: halve the window when congestion/loss is seen;
            # otherwise increase by one packet per RTT.
            if total_loss_events:
                cwnd = max(2.0, cwnd * 0.5)
            else:
                cwnd = min(float(max_window), cwnd + 1.0)
            pacing_pps = cwnd / dt

        budget = max(1, min(int(cwnd), int(pacing_pps * dt)))
        offered: list[tuple[int, bool]] = []
        while retries and len(offered) < budget:
            offered.append((retries.pop(0), True))
        while next_packet_id < total_packets and len(offered) < budget:
            offered.append((next_packet_id, False))
            next_packet_id += 1

        step_attempts = 0
        step_losses = 0
        for packet_id, is_retry in offered:
            step_attempts += 1
            transmission_attempts += 1
            if is_retry:
                retransmissions += 1
            if len(fifo) >= scenario.queue_capacity_packets:
                retries.append(packet_id)
                step_losses += 1
                continue
            fifo.append((packet_id, time_s))

        # Serve FIFO packets at the link rate. The residual queue persists.
        service_count = min(service_per_epoch, len(fifo))
        delivered_this_epoch = 0
        for index in range(service_count):
            packet_id, enqueue_time = fifo[index]
            if rng.random() < scenario.random_loss_rate:
                retries.append(packet_id)
                step_losses += 1
                continue
            if packet_id not in delivered:
                delivered.add(packet_id)
                delivered_this_epoch += 1
                queue_wait_ms = max(0.0, time_s - enqueue_time) * 1000.0
                serialization_wait_ms = (index / scenario.capacity_pps) * 1000.0
                delivered_delays.append(
                    scenario.base_rtt_ms + queue_wait_ms + serialization_wait_ms
                )
        del fifo[:service_count]

        # Keep the immediate previous interval as the controller's loss input.
        total_loss_events = step_losses
        previous_loss_rate = min(1.0, step_losses / max(1, step_attempts))
        previous_queue = len(fifo)
        if mode == "ppo":
            assert agent is not None
            next_observation = Observation(
                loss_rate=previous_loss_rate,
                delay_ms=scenario.base_rtt_ms + (previous_queue / scenario.capacity_pps) * 1000.0,
                baseline_delay_ms=scenario.base_rtt_ms,
                congestion=min(1.0, previous_queue / scenario.queue_capacity_packets),
            )
            # Reward useful service while penalizing queue buildup, loss, and
            # delay inflation across the controlled path scenarios.
            delay_inflation = max(0.0, next_observation.delay_ms /
                                  next_observation.baseline_delay_ms - 1.0)
            reward = (delivered_this_epoch / service_per_epoch
                      - 0.35 * next_observation.congestion
                      - 1.20 * previous_loss_rate
                      - 0.20 * min(1.0, delay_inflation / 2.0))
            if training:
                agent.observe_reward(reward, next_observation, done=len(delivered) >= total_packets)
        time_s += dt

    completed = len(delivered) == total_packets
    elapsed_s = max(time_s, dt)
    useful_bits = len(delivered) * packet_size_bytes * 8
    throughput_mbps = useful_bits / elapsed_s / 1_000_000
    average_delay_ms = sum(delivered_delays) / len(delivered_delays) if delivered_delays else 0.0
    return SimulationResult(
        scenario=scenario.name,
        mode=mode,
        seed=seed,
        throughput_mbps=throughput_mbps,
        packet_delivery_ratio=len(delivered) / total_packets,
        average_delay_ms=average_delay_ms,
        retransmissions=retransmissions,
        packets_sent=transmission_attempts,
        elapsed_seconds=elapsed_s,
        completed=completed,
    )


if __name__ == "__main__":
    policy_candidates = [
        Path(__file__).resolve().parent.parent / "results" / "ppo_policy.json",
        Path("results/ppo_policy.json"),
        Path("ppo_policy.json"),
    ]
    policy_file = next((p for p in policy_candidates if p.exists()), None)
    demo_agent = PPOController.load(policy_file) if policy_file else PPOController()
    for test_scenario in SCENARIOS:
        for test_mode in ("reno", "heuristic", "ppo"):
            result = simulate(test_scenario, test_mode, seed=7,
                              ppo_agent=demo_agent if test_mode == "ppo" else None)
            print(result)
