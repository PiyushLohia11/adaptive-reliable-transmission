"""Dependency-free clipped PPO policy for adaptive reliable transmission.

This educational implementation uses small actor/critic MLPs and categorical
actions. It follows clipped PPO with generalized advantage estimation (GAE).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class Observation:
    loss_rate: float
    delay_ms: float
    baseline_delay_ms: float
    congestion: float


class PPOController:
    """PPO actor-critic mapping network feedback to sender-rate actions."""

    ACTIONS = ((0.93, 0.93), (1.00, 1.00), (1.07, 1.07))
    ACTION_NAMES = ("decrease", "hold", "increase")
    HIDDEN = 12

    def __init__(self, seed: int = 0, gamma: float = 0.97, gae_lambda: float = 0.92,
                 clip: float = 0.20, actor_lr: float = 0.0015,
                 critic_lr: float = 0.004, entropy_coef: float = 0.015) -> None:
        self.rng = random.Random(seed)
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.actor_lr, self.critic_lr, self.entropy_coef = actor_lr, critic_lr, entropy_coef
        self.actor = self._new_network(3)
        self.critic = self._new_network(1)
        self._adam = {"actor": self._zero_tree(self.actor), "critic": self._zero_tree(self.critic)}
        self._adam_v = {"actor": self._zero_tree(self.actor), "critic": self._zero_tree(self.critic)}
        self._adam_t = {"actor": 0, "critic": 0}
        self.pending: tuple[list[float], int, float, float] | None = None
        self.rollout: list[dict[str, Any]] = []

    @classmethod
    def features(cls, observation: Observation) -> list[float]:
        if not 0 <= observation.loss_rate <= 1 or not 0 <= observation.congestion <= 1:
            raise ValueError("loss rate and congestion must be in [0, 1]")
        if observation.baseline_delay_ms <= 0 or observation.delay_ms < 0:
            raise ValueError("delay must be nonnegative and baseline delay positive")
        inflation = max(0.0, observation.delay_ms / observation.baseline_delay_ms - 1.0)
        return [observation.loss_rate, min(2.0, inflation) / 2.0, observation.congestion]

    def _new_network(self, outputs: int) -> dict[str, Any]:
        scale = 0.35
        return {
            "w1": [[self.rng.gauss(0, scale / math.sqrt(3)) for _ in range(3)] for _ in range(self.HIDDEN)],
            "b1": [0.0] * self.HIDDEN,
            "w2": [[self.rng.gauss(0, scale / math.sqrt(self.HIDDEN)) for _ in range(self.HIDDEN)] for _ in range(outputs)],
            "b2": [0.0] * outputs,
        }

    @staticmethod
    def _zero_tree(tree: Any) -> Any:
        if isinstance(tree, dict):
            return {key: PPOController._zero_tree(value) for key, value in tree.items()}
        return [PPOController._zero_tree(item) for item in tree] if isinstance(tree, list) else 0.0

    @staticmethod
    def _forward(net: dict[str, Any], x: list[float]) -> tuple[list[float], list[float]]:
        hidden = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + bias)
                  for row, bias in zip(net["w1"], net["b1"])]
        out = [sum(w * h for w, h in zip(row, hidden)) + bias
               for row, bias in zip(net["w2"], net["b2"])]
        return out, hidden

    def _policy_value(self, x: list[float]) -> tuple[list[float], float]:
        logits, _ = self._forward(self.actor, x)
        peak = max(logits)
        exps = [math.exp(max(-40.0, z - peak)) for z in logits]
        total = sum(exps)
        probs = [max(1e-8, value / total) for value in exps]
        values, _ = self._forward(self.critic, x)
        return probs, values[0]

    def act(self, observation: Observation, deterministic: bool = False) -> tuple[int, float, float]:
        x = self.features(observation)
        probs, value = self._policy_value(x)
        if deterministic:
            action = max(range(len(probs)), key=lambda i: probs[i])
        else:
            draw, acc = self.rng.random(), 0.0
            action = len(probs) - 1
            for i, prob in enumerate(probs):
                acc += prob
                if draw <= acc:
                    action = i
                    break
        self.pending = (x, action, math.log(probs[action]), value)
        cwnd_multiplier, pacing_multiplier = self.ACTIONS[action]
        return action, cwnd_multiplier, pacing_multiplier

    def observe_reward(self, reward: float, next_observation: Observation, done: bool) -> None:
        if self.pending is None:
            return
        x, action, log_prob, value = self.pending
        next_x = self.features(next_observation)
        self.rollout.append({"x": x, "action": action, "logp": log_prob,
                             "value": value, "reward": float(reward),
                             "next_x": next_x, "done": bool(done)})
        self.pending = None

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        peak = max(logits)
        exps = [math.exp(max(-40.0, z - peak)) for z in logits]
        total = sum(exps)
        return [max(1e-8, value / total) for value in exps]

    @staticmethod
    def _add_network_grad(net: dict[str, Any], grads: dict[str, Any],
                          x: list[float], output_grad: list[float]) -> None:
        _, hidden = PPOController._forward(net, x)
        hidden_grad = [0.0] * len(hidden)
        for o, grad in enumerate(output_grad):
            grads["b2"][o] += grad
            for h, hv in enumerate(hidden):
                grads["w2"][o][h] += grad * hv
                hidden_grad[h] += net["w2"][o][h] * grad
        for h, hv in enumerate(hidden):
            pre_grad = hidden_grad[h] * (1.0 - hv * hv)
            grads["b1"][h] += pre_grad
            for i, xi in enumerate(x):
                grads["w1"][h][i] += pre_grad * xi

    @staticmethod
    def _adam_tree(params: Any, grads: Any, m: Any, v: Any, t: int, lr: float) -> tuple[Any, Any, Any]:
        if isinstance(params, dict):
            updated, new_m, new_v = {}, {}, {}
            for key in params:
                updated[key], new_m[key], new_v[key] = PPOController._adam_tree(
                    params[key], grads[key], m[key], v[key], t, lr)
            return updated, new_m, new_v
        if isinstance(params, list):
            results = [PPOController._adam_tree(p, g, mm, vv, t, lr)
                       for p, g, mm, vv in zip(params, grads, m, v)]
            return ([r[0] for r in results], [r[1] for r in results], [r[2] for r in results])
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mn = beta1 * m + (1 - beta1) * grads
        vn = beta2 * v + (1 - beta2) * grads * grads
        pn = params - lr * (mn / (1 - beta1 ** t)) / (math.sqrt(vn / (1 - beta2 ** t)) + eps)
        return pn, mn, vn

    @staticmethod
    def _empty_grads(net: dict[str, Any]) -> dict[str, Any]:
        return PPOController._zero_tree(net)

    def update(self, epochs: int = 4) -> float:
        """Apply clipped PPO updates to the current episode rollout."""
        if not self.rollout:
            return 0.0
        samples = self.rollout
        advantages = [0.0] * len(samples)
        gae = 0.0
        for i in range(len(samples) - 1, -1, -1):
            row = samples[i]
            next_value = 0.0 if row["done"] else self._forward(self.critic, row["next_x"])[0][0]
            delta = row["reward"] + self.gamma * next_value * (not row["done"]) - row["value"]
            gae = delta + self.gamma * self.gae_lambda * (not row["done"]) * gae
            advantages[i] = gae
        mean_adv = sum(advantages) / len(advantages)
        var_adv = sum((a - mean_adv) ** 2 for a in advantages) / max(1, len(advantages))
        scale_adv = math.sqrt(var_adv) + 1e-8
        advantages = [(a - mean_adv) / scale_adv for a in advantages]
        returns = [a + row["value"] for a, row in zip(advantages, samples)]

        for _ in range(epochs):
            order = list(range(len(samples)))
            self.rng.shuffle(order)
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                actor_grad, critic_grad = self._empty_grads(self.actor), self._empty_grads(self.critic)
                for idx in batch:
                    row, adv, target = samples[idx], advantages[idx], returns[idx]
                    n = len(batch)
                    probs, value = self._policy_value(row["x"])
                    logp = math.log(probs[row["action"]])
                    ratio = math.exp(max(-20.0, min(20.0, logp - row["logp"])))
                    clipped_ratio = min(max(ratio, 1.0 - self.clip), 1.0 + self.clip)
                    use_gradient = not ((adv >= 0 and ratio > 1.0 + self.clip) or
                                        (adv < 0 and ratio < 1.0 - self.clip))
                    dlogp = -ratio * adv / n if use_gradient else 0.0
                    entropy = -sum(p * math.log(p) for p in probs)
                    logits_grad = []
                    for j, p in enumerate(probs):
                        policy_grad = dlogp * ((1.0 if j == row["action"] else 0.0) - p)
                        entropy_grad = self.entropy_coef * p * (math.log(p) + entropy) / n
                        logits_grad.append(policy_grad + entropy_grad)
                    self._add_network_grad(self.actor, actor_grad, row["x"], logits_grad)
                    self._add_network_grad(self.critic, critic_grad, row["x"],
                                           [0.5 * 2.0 * (value - target) / n])
                self._adam_t["actor"] += 1
                self._adam_t["critic"] += 1
                self.actor, self._adam["actor"], self._adam_v["actor"] = self._adam_tree(
                    self.actor, actor_grad, self._adam["actor"], self._adam_v["actor"],
                    self._adam_t["actor"], self.actor_lr)
                self.critic, self._adam["critic"], self._adam_v["critic"] = self._adam_tree(
                    self.critic, critic_grad, self._adam["critic"], self._adam_v["critic"],
                    self._adam_t["critic"], self.critic_lr)
        mean_reward = sum(row["reward"] for row in samples) / len(samples)
        self.rollout = []
        return mean_reward

    def save(self, path: str | Path) -> None:
        data = {"algorithm": "clipped_ppo_actor_critic", "actions": self.ACTION_NAMES,
                "action_multipliers": self.ACTIONS, "gamma": self.gamma,
                "gae_lambda": self.gae_lambda, "clip": self.clip,
                "actor_lr": self.actor_lr, "critic_lr": self.critic_lr,
                "entropy_coef": self.entropy_coef, "actor": self.actor, "critic": self.critic}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PPOController":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("algorithm") != "clipped_ppo_actor_critic":
            raise ValueError("unrecognized policy format")
        agent = cls(seed=seed, gamma=data["gamma"], gae_lambda=data["gae_lambda"],
                    clip=data["clip"], actor_lr=data["actor_lr"],
                    critic_lr=data["critic_lr"], entropy_coef=data["entropy_coef"])
        agent.actor, agent.critic = data["actor"], data["critic"]
        return agent
