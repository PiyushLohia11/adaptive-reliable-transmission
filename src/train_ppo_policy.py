"""Train a clipped PPO actor-critic policy on simulation episodes."""

from __future__ import annotations

import csv
from pathlib import Path

try:
    from .network_simulator import SCENARIOS, simulate
    from .ppo_controller import PPOController
except ImportError:
    from network_simulator import SCENARIOS, simulate
    from ppo_controller import PPOController


ROOT_DIR = Path(__file__).resolve().parent.parent
OUT = ROOT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)
EPISODES = 120
TRAINING_PACKETS = 2000


def main() -> PPOController:
    agent = PPOController(seed=2026)
    rows = []
    for episode in range(EPISODES):
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        result = simulate(
            scenario,
            "ppo",
            seed=10000 + episode,
            total_packets=TRAINING_PACKETS,
            ppo_agent=agent,
            training=True,
        )
        mean_reward = agent.update(epochs=4)
        rows.append({
            "episode": episode + 1,
            "scenario": scenario.name,
            "mean_reward": f"{mean_reward:.6f}",
            "throughput_mbps": f"{result.throughput_mbps:.6f}",
            "average_delay_ms": f"{result.average_delay_ms:.6f}",
            "retransmissions": result.retransmissions,
            "completed": result.completed,
        })
    agent.save(OUT / "ppo_policy.json")
    with (OUT / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Trained PPO for {EPISODES} episodes ({TRAINING_PACKETS} packets each)")
    print(f"Saved {OUT / 'ppo_policy.json'} and {OUT / 'training_history.csv'}")
    return agent


if __name__ == "__main__":
    main()
