# AI-Driven Adaptive Reliable Transmission for Lossy Networks

## Abstract

This project evaluates reliable packet transfer in a Python teaching simulator using three sender controllers: simplified Reno-style AIMD, a hand-designed rule heuristic, and a custom clipped Proximal Policy Optimization (PPO) actor-critic. The learned policy observes packet loss, delay inflation, and queue occupancy, then adjusts its congestion window and pacing rate. It trains for 120 episodes on seeds separate from the ten held-out evaluation seeds used in each of four scenarios. Across the 120 held-out runs, PPO improved goodput over the heuristic in all four scenarios, with the largest gain at high delay. These goodput gains came with higher latency, and PPO generated substantially more retransmissions in the high-delay case. The experiment shows a functioning learning-based controller and measurable trade-offs, not universal superiority or real-network performance.

## 1. Objective and scope

The objective is to develop and experimentally validate an adaptive reliable-transmission approach under controlled loss, delay, and congestion. The faculty outline covers deep reinforcement learning for XQUIC, queue management, and multipath routing. This semester prototype implements a smaller single-flow sender and FIFO bottleneck model. Its learning algorithm is clipped PPO with compact neural actor and critic networks. The implementation uses Python's standard library and is custom code, not Stable-Baselines3. A heuristic adaptive policy remains as a non-learning comparison, alongside a simplified Reno-style AIMD baseline.

## 2. Method

### 2.1 PPO controller

The policy state contains three normalized values: recent interval loss rate, delay inflation relative to unloaded RTT, and normalized queue occupancy. A 12-unit tanh hidden layer feeds a categorical actor with three actions and a scalar critic. The actor can represent continuous variation across the network states, rather than assigning a fixed action to each discrete state.

- Loss rate is normalized to `[0,1]`.
- Delay inflation is `max(0, measured_RTT / baseline_RTT − 1)`, clipped at 2 and scaled to `[0,1]`.
- Queue occupancy is normalized to `[0,1]` of FIFO capacity.

The actor selects a paired sender action: multiply both congestion window and pacing by 0.93, 1.00, or 1.07 (decrease, hold, increase). It samples from the categorical policy during training and chooses its highest-probability action during evaluation. PPO maximizes the clipped policy objective:

`L_clip = E[min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t)]`

Here `r_t` is the ratio of new to old action probability, the clipping parameter is `ε = 0.20`, and `A_t` is a generalized advantage estimate. The discount is `γ = 0.97` and GAE parameter is `λ = 0.92`. A 12-hidden-unit critic estimates returns. The reward for each epoch is:

`r = delivered / link_service − 0.35 × queue_fraction − 1.20 × loss_rate − 0.20 × min(1, delay_inflation / 2)`

This rewards useful link utilization and penalizes queue buildup, loss, and delay inflation. PPO uses four update epochs per episode with minibatches of 64, actor learning rate `0.0015`, critic learning rate `0.004`, and entropy coefficient `0.015`. The action size, reward weights, architecture, and training episode count are design choices. The implementation and exported network weights are in `ppo_controller.py` and `ppo_policy.json`.

### 2.2 Training and evaluation separation

PPO trains for 120 episodes of 2000 packets, rotating evenly across the four scenarios. Training uses seeds 10000–10119. The final frozen policy is evaluated using seeds 200–209, transferring 5000 packets in each run. These seed ranges do not overlap. Each scenario and seed is tested in all three modes, for 120 held-out runs. The heuristic does not learn; it remains a hand-designed baseline. Reno-style AIMD is also not a complete TCP implementation.

### 2.3 Simulator and sender modes

The simulator advances in epochs equal to the baseline RTT. A sender offers new or retry packets according to its congestion window and pacing rate. A finite FIFO queue serves packets at the configured link rate. Packets can be dropped by independent random link loss or queue overflow, then retried in a later epoch. Unique packet IDs support useful-goodput and delivery measurements.

The Reno-style baseline increases the window by one packet per epoch without detected loss and halves it after a loss interval. The heuristic consumes the same loss, delay, and queue feedback and adjusts window/pacing with fixed equations. PPO selects one of three paired window and pacing multipliers. All modes share the queue, workload, path settings, and retry mechanism.

### 2.4 Scenarios and metrics

Each run transfers 5000 unique 1200-byte packets.

| Scenario | Random link loss | Baseline RTT | Service capacity | Queue capacity |
|---|---:|---:|---:|---:|
| Clean link | 0% | 20 ms | 1000 packets/s | 100 packets |
| Wireless loss | 8% | 20 ms | 1000 packets/s | 100 packets |
| High delay | 2% | 100 ms | 1000 packets/s | 100 packets |
| Congested bottleneck | 5% | 40 ms | 500 packets/s | 24 packets |

Reported metrics are useful throughput (unique payload bits divided by simulated elapsed time), unique packet delivery ratio, mean end-to-end delay for delivered packets, and retransmission count. Summary tables report mean and sample standard deviation over ten held-out seeds.

## 3. Results

All 120 evaluation runs completed the 5000-packet workload. PDR is therefore 100% for every controller and scenario because the simulator retries until delivery; it does not distinguish performance here.

| Scenario | Controller | Throughput Mbps | Delivery ratio | Mean delay ms | Retransmissions |
|---|---|---:|---:|---:|---:|
| Clean link | Reno-style | 9.562 ± 0.000 | 1.000 | 90.809 ± 0.000 | 122.0 ± 0.0 |
| Clean link | Rule heuristic | 9.023 ± 0.000 | 1.000 | 32.743 ± 0.000 | 0.0 ± 0.0 |
| Clean link | PPO | 9.524 ± 0.000 | 1.000 | 36.841 ± 0.000 | 0.0 ± 0.0 |
| Wireless loss | Reno-style | 1.893 ± 0.039 | 1.000 | 22.100 ± 0.058 | 436.7 ± 13.1 |
| Wireless loss | Rule heuristic | 8.430 ± 0.045 | 1.000 | 31.313 ± 0.071 | 436.7 ± 13.1 |
| Wireless loss | PPO | 8.705 ± 0.042 | 1.000 | 35.131 ± 0.238 | 436.7 ± 13.1 |
| High delay | Reno-style | 0.817 ± 0.052 | 1.000 | 104.743 ± 0.289 | 101.6 ± 10.2 |
| High delay | Rule heuristic | 5.412 ± 0.050 | 1.000 | 136.606 ± 0.213 | 416.6 ± 7.4 |
| High delay | PPO | 7.208 ± 0.056 | 1.000 | 144.230 ± 0.047 | 1791.5 ± 29.7 |
| Congested bottleneck | Reno-style | 1.223 ± 0.046 | 1.000 | 45.581 ± 0.288 | 263.7 ± 13.7 |
| Congested bottleneck | Rule heuristic | 4.367 ± 0.015 | 1.000 | 60.689 ± 0.065 | 271.1 ± 12.9 |
| Congested bottleneck | PPO | 4.523 ± 0.012 | 1.000 | 63.796 ± 0.082 | 323.8 ± 12.3 |

Relative to the heuristic, PPO goodput changed by +5.6% on the clean link, +3.3% with wireless loss, +33.2% at high delay, and +3.6% at the congested bottleneck. It increased mean delay in all four cases by 12.5%, 12.2%, 5.6%, and 5.1%, respectively. Retransmissions matched the heuristic under clean and wireless conditions, were 330.1% higher at high delay, and 19.4% higher under congestion. The goodput advantage comes with a clear latency cost and a substantial retry cost at high delay.

### 3.1 Figures

Bars show means and whiskers show ±1 sample standard deviation across ten held-out seeds.

![Throughput comparison](../figures/throughput_mbps.svg)

![Packet delivery ratio comparison](../figures/packet_delivery_ratio.svg)

![End-to-end delay comparison](../figures/average_delay_ms.svg)

![Retransmission comparison](../figures/retransmissions.svg)

## 4. Discussion

The implementation contains a learning component: clipped PPO updates a neural actor and critic from experience during training episodes, and the saved policy runs greedily during evaluation. Its held-out behavior is measurable and condition-dependent. PPO raises goodput over the heuristic in all four configured scenarios, especially the high-delay case, but also increases delay in all four and generates many more retries under high delay. The experiment therefore exposes a performance trade-off and leaves room to improve reward calibration or action design.

The experiment supports the claim that a PPO-based adaptive sender was implemented and evaluated against both conventional and non-learning adaptive controls. It does not establish that PPO is the best policy, that reward weights generalize beyond these four scenarios, or that results carry over to TCP, QUIC, or a physical network. A further study should use more training variation, more held-out seeds, competing flows, burst loss, and a recognized network simulator.

## 5. Reproduction and limitations

From this folder, run:

```powershell
python verify_project.py
python run_experiments.py
```

The experiment runner retrains the policy, writes `ppo_policy.json` and `training_history.csv`, then creates `replicate_results.csv`, `summary_results.csv`, and four SVG plots. The PPO policy is frozen during held-out evaluation. The simulator uses one flow, one FIFO bottleneck, independent random drops, simplified RTT epochs, and immediate simulator knowledge of packet losses for later retries. It does not model ACK loss, retransmission timeouts, cross traffic, multiple competing flows, burst errors, FEC, or real TCP/QUIC. All runs eventually deliver every packet, so this does not measure deadline reliability or prolonged outages. Results are simulated, not real-network measurements.

## 6. Conclusion

The project provides a working Python transmission framework with Reno-style, rule-based, and PPO sender modes. PPO trains for 120 episodes and is evaluated on disjoint seeds. It improves goodput over the hand-designed heuristic across the four configured scenarios, while raising mean delay and, in the high-delay scenario, retransmissions. The findings meet the project objective of implementing and experimentally validating an AI-driven adaptive method in a controlled simulation, with the limitations stated above.

## References

The following references were supplied by the faculty. The congestion-control, queue-management, and RL papers most directly motivate this scope; the broader list is retained for project context.

1. “DNCCQ-PPO: A dynamic network congestion control algorithm based on deep reinforcement learning for XQUIC,” *Journal of Network and Computer Applications*, vol. 245, Art. 104371, 2026. [https://doi.org/10.1016/j.jnca.2025.104371](https://doi.org/10.1016/j.jnca.2025.104371)
2. H. Ma, D. Xu, and X. Wang, “Congestion control algorithm-aware queue management,” *Computer Networks*, vol. 276, Art. 111975, 2026. [https://doi.org/10.1016/j.comnet.2025.111975](https://doi.org/10.1016/j.comnet.2025.111975)
3. J. Zhang, J. Zhou, and J. Yuan, “Active queue management based on fuzzy logic and reinforcement learning,” *Computer Networks*, vol. 282, Art. 112272, 2026. [https://doi.org/10.1016/j.comnet.2026.112272](https://doi.org/10.1016/j.comnet.2026.112272)
4. M. P. Toopchinezhad and M. Ahmadi, “Machine learning approaches for active queue management: A survey, taxonomy, and future directions,” *Computer Networks*, vol. 262, Art. 111174, 2025. [https://doi.org/10.1016/j.comnet.2025.111174](https://doi.org/10.1016/j.comnet.2025.111174)
5. S. K. Pattiwar, P. Saxena, and Ö. Alay, “PRISM: Proximal policy optimization with deep Reinforcement learning for Intelligent Scheduling in Multipath QUIC under heterogeneous and hybrid 5G/B5G-satellite networks,” *Computer Communications*, vol. 252, Art. 108523, 2026. [https://doi.org/10.1016/j.comcom.2026.108523](https://doi.org/10.1016/j.comcom.2026.108523)
6. M. Luglio, C. Roseti, D. Verde, and F. Zampognaro, “QUIC Wave: Bringing bursts at application level to efficiently face new communication scenarios,” *Computer Networks*, vol. 281, Art. 112185, 2026. [https://doi.org/10.1016/j.comnet.2026.112185](https://doi.org/10.1016/j.comnet.2026.112185)
7. L. P. Aguirre S., Y. Shen, and M. Guo, “LCA: Deep Reinforcement Learning-Based Congestion Avoidance Routing Model in SDN,” *Computer Networks*, vol. 268, Art. 111371, 2025. [https://doi.org/10.1016/j.comnet.2025.111371](https://doi.org/10.1016/j.comnet.2025.111371)
8. “MTEAL: Network routing optimization of SD-WAN traffic engineering integrating multi-dimensional QoS metrics,” *Journal of Network and Computer Applications*, vol. 242, Art. 104272, 2025. [https://doi.org/10.1016/j.jnca.2025.104272](https://doi.org/10.1016/j.jnca.2025.104272)
9. R. Aiyshwariya Devi et al., “Traffic prediction and mitigation in 6G software-defined networks using bobcat Bi-GRU and reinforcement learning based on proximal policy optimization,” *Knowledge-Based Systems*, vol. 328, Art. 114243, 2025. [https://doi.org/10.1016/j.knosys.2025.114243](https://doi.org/10.1016/j.knosys.2025.114243)
10. Z. Chen et al., “Multi-policy reinforcement learning for network resource allocation with periodic behaviors,” *Computer Networks*, vol. 272, Art. 111645, 2025. [https://doi.org/10.1016/j.comnet.2025.111645](https://doi.org/10.1016/j.comnet.2025.111645)
