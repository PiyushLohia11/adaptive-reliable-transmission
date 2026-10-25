# Person 1: PPO adaptive sender

## Algorithm

This implementation uses Proximal Policy Optimization (PPO), an on-policy reinforcement-learning method. A neural actor chooses transmission changes and a neural critic estimates value. During training, the sender samples actions, receives simulated feedback, computes advantages, and updates the networks using PPO's clipped policy objective. The policy is saved and runs deterministically on held-out tests.

This is custom PPO implemented with Python's standard library. It does not use Stable-Baselines3, Gymnasium, PyTorch, or NumPy, so the extracted project runs without external dependencies. The implementation is an educational PPO actor-critic, not a production transport controller or a deep PPO reproduction from the cited papers.

## State and action

The actor observes three normalized values:

- Recent packet loss rate in `[0, 1]`.
- Delay inflation above baseline RTT, clipped at 2× and scaled to `[0, 1]`.
- Queue occupancy divided by queue capacity in `[0, 1]`.

The actor and critic each use a 12-unit tanh hidden layer. The categorical actor selects one joint congestion-window and pacing action:

| Action | Window multiplier | Pacing multiplier |
|---|---:|---:|
| Decrease | 0.93 | 0.93 |
| Hold | 1.00 | 1.00 |
| Increase | 1.07 | 1.07 |

The simulator bounds the resulting window and pacing rate.

## PPO training

PPO trains for 120 episodes of 2000 packets each, cycling through all four conditions. Training seeds are 10000–10119; held-out seeds are 200–209. The policy has four PPO update epochs per episode, minibatches of 64, clipping parameter 0.20, discount factor 0.97, GAE lambda 0.92, actor learning rate 0.0015, critic learning rate 0.004, and entropy coefficient 0.015. Evaluation freezes the networks and chooses the action with the largest policy probability.

The per-epoch reward is:

`delivered_packets / link_service_packets − 0.35 × queue_fraction − 1.20 × loss_rate − 0.20 × min(1, delay_inflation / 2)`

This reward values link use and penalizes queue buildup, loss, and delay. Reward weights are experiment design choices and affect the learned trade-off.

## Implemented files

- `src/ppo_controller.py`: actor-critic networks, rollout buffer, GAE, clipped PPO loss, optimizer, and policy save/load.
- `src/train_ppo_policy.py`: training episodes and policy export.
- `src/network_simulator.py`: state and reward integration with FIFO and retransmission simulation.
- `results/ppo_policy.json`: learned actor/critic weights from the last experiment run.
- `results/training_history.csv`: episode-level training records.
- `src/adaptive_controller.py`: retained rule-based adaptive baseline.

## Evaluation against the heuristic

In ten held-out seeds per scenario, PPO goodput changed by +5.6% on the clean link, +3.3% under wireless loss, +33.2% under high delay, and +3.6% under bottleneck congestion relative to the rule heuristic. Mean delay was also higher in every scenario. Retransmissions matched the heuristic in clean and wireless conditions, but were 330.1% higher under high delay and 19.4% higher under congestion. This makes the throughput gain visible while exposing a substantial high-delay reliability cost. Since the simulator retries until delivery, PDR is 100% for all modes and does not distinguish them.

## Reproduce

From the extracted project folder, run `python run_experiments.py`. It trains a fresh PPO model, writes the policy and training history, evaluates all three modes on held-out seeds, and regenerates CSV results and plots.
