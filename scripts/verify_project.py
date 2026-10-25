"""Fast checks for controller bounds, directional behavior, and simulation repeatability."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from adaptive_controller import AdaptiveController, NetworkMetrics
    from network_simulator import SCENARIOS, simulate
    from ppo_controller import Observation, PPOController
except ImportError:
    from src.adaptive_controller import AdaptiveController, NetworkMetrics
    from src.network_simulator import SCENARIOS, simulate
    from src.ppo_controller import Observation, PPOController


def main() -> None:
    controller = AdaptiveController()
    clean = controller.update(NetworkMetrics(0.0, 20, 20, 0.0))
    assert clean.cwnd_multiplier > 1.0
    controller.reset()
    random_loss = controller.update(NetworkMetrics(0.10, 21, 20, 0.02))
    assert random_loss.redundancy_fraction > 0
    assert random_loss.cwnd_multiplier > 0.9
    controller.reset()
    severe = controller.update(NetworkMetrics(0.15, 140, 20, 0.95))
    assert severe.cwnd_multiplier < clean.cwnd_multiplier
    assert severe.pacing_multiplier < clean.pacing_multiplier
    for invalid in (
        NetworkMetrics(-0.1, 20, 20, 0.0),
        NetworkMetrics(0.1, 20, 20, 1.2),
        NetworkMetrics(0.1, 20, 0, 0.2),
    ):
        try:
            AdaptiveController().update(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid metrics accepted: {invalid}")

    ppo = PPOController(seed=3)
    calm = Observation(0.0, 20, 20, 0.0)
    stressed = Observation(0.12, 100, 20, 0.8)
    assert PPOController.features(calm) != PPOController.features(stressed)
    action, cwnd_multiplier, pacing_multiplier = ppo.act(calm, deterministic=True)
    assert action in (0, 1, 2)
    assert 0.90 <= cwnd_multiplier <= 1.10 and 0.90 <= pacing_multiplier <= 1.10
    ppo.observe_reward(0.5, stressed, done=True)
    ppo.update(epochs=1)
    assert not ppo.rollout

    for scenario in SCENARIOS:
        for mode in ("reno", "heuristic", "ppo"):
            kwargs = {"ppo_agent": PPOController(seed=71)} if mode == "ppo" else {}
            a = simulate(scenario, mode, seed=913, **kwargs)
            kwargs = {"ppo_agent": PPOController(seed=71)} if mode == "ppo" else {}
            b = simulate(scenario, mode, seed=913, **kwargs)
            assert a == b, "same seed must reproduce the same run"
            assert a.completed and a.packet_delivery_ratio == 1.0
            assert a.throughput_mbps > 0 and a.average_delay_ms >= scenario.base_rtt_ms
    print("PASS: heuristic and PPO validation, deterministic modes, and full delivery")


if __name__ == "__main__":
    main()
