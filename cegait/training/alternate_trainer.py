"""Algorithm 1: warm-up, detached attacker steps, differentiable protector step."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..baselines import make_protector
from ..models.query_set_adversary import QuerySetAdversary
from ..losses.utility_loss import batch_hard_triplet, quantization_loss
from ..losses.marginal_linkage_gain import frozen_adversary, marginal_linkage_gain, confusion_loss
from ..losses.exposure_group_risk import exposure_group_risk
from ..protocols.domain_key_schedule import DomainKeySchedule
from ..protocols.history_episode_builder import HistoryEpisodeBuilder, encode_episodes
from ..data import write_json
from ..checkpoint import read_checkpoint


def load_protector(checkpoint, device="cpu"):
    saved = read_checkpoint(checkpoint, device, "protector")
    model = make_protector(saved.get("method", "cegait"), **saved.get("model_options", {})).to(device)
    model.load_state_dict(saved["protector"], strict=True)
    return model.eval(), saved


class AlternateTrainer:
    def __init__(self, store, config, output, device="cpu"):
        self.store, self.config, self.device = store, config, device
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.context_rng = torch.Generator().manual_seed(self.seed)
        self.method = config.get("method", "cegait")
        self.model_options = config.get("model", {})
        self.protector = make_protector(self.method, **self.model_options).to(device)
        self.adversary = QuerySetAdversary().to(device)
        self.train_options = config["training"]
        options = self.train_options
        self.gopt = torch.optim.AdamW(self.protector.parameters(), lr=options["protector_lr"], weight_decay=options["weight_decay"])
        self.aopt = torch.optim.AdamW(self.adversary.parameters(), lr=options["attacker_lr"], weight_decay=options["weight_decay"])
        self.builder = HistoryEpisodeBuilder(store, self.seed)
        self.groups = []
        for family in config["protocol"]["families"]:
            for length in config["protocol"]["lengths"]:
                if len(self.builder.eligible("generator", family, length, cohort_length=length)) >= 2:
                    self.groups.append((family, length))
        if not self.groups:
            raise ValueError("No supported training exposure groups")
        self.plans = {(g, m): self.builder.build("generator", g, m, draws=2, cohort_length=m) for g, m in self.groups}
        self.best, self.start = float("inf"), 0
        write_json(self.output / "config.json", config)

    def sample_groups(self):
        budget = self.train_options["episodes"]
        if budget < 2 or budget % 2:
            raise ValueError("An even number of at least two linkage episodes is required")
        # Distribute balanced +/- pairs across groups; every group is represented
        # at the paper batch size of 32 for the 12 CASIA-B groups.
        selected = list(self.groups)
        if budget // 2 < len(selected):
            indices = self.rng.choice(len(selected), budget // 2, replace=False)
            selected = [selected[i] for i in indices]
        pairs = [budget // (2 * len(selected))] * len(selected)
        for i in range(budget // 2 - sum(pairs)):
            pairs[i] += 1
        result = []
        for group, count in zip(selected, pairs):
            plan = self.plans[group]
            chosen = self.rng.integers(len(plan) // 2, size=count) * 2
            indices = np.stack([chosen, chosen + 1], 1).reshape(-1)
            result.append((group, plan.subset(indices)))
        return result

    def utility(self, iteration, role="generator"):
        options = self.train_options
        features, labels = self.store.utility_batch(role, options["utility_identities"], options["utility_clips"],
                                                    self.rng, self.device, getattr(self.protector, "feature_level", False))
        key = DomainKeySchedule(self.seed, f"utility/{role}").vectors(iteration, ["active"], self.device)
        output = self.protector(features, key)
        utility = batch_hard_triplet(output["template"], labels)
        if self.method == "priv_ihgr":
            other_key = DomainKeySchedule(self.seed, f"utility/{role}").vectors(iteration, ["other"], self.device)
            other = self.protector(features, other_key)
            utility = utility + self.protector.cross_key_loss(output["template"], other["template"])
        return utility, quantization_loss(output["continuous"], output["template"])

    def attack_step(self, iteration, substep):
        self.adversary.train()
        self.aopt.zero_grad(set_to_none=True)
        losses = []
        for (_, _), episodes in self.sample_groups():
            with torch.no_grad():
                h, c, y, _, _ = encode_episodes(self.protector, self.store, episodes,
                                                f"{iteration}/{substep}", "training", self.seed, self.device)
            full = F.binary_cross_entropy_with_logits(self.adversary(h.detach(), c.detach()), y)
            mask = torch.zeros(h.shape[:2], dtype=torch.bool, device=self.device)
            for row in range(len(h)):
                size = int(self.rng.integers(1, h.shape[1] + 1))
                mask[row, self.rng.choice(h.shape[1], size, replace=False)] = True
            partial = F.binary_cross_entropy_with_logits(self.adversary(h.detach(), c.detach(), mask), y)
            losses.append((full + partial) / 2)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), 5)
        self.aopt.step()
        return float(loss.detach())

    def protector_step(self, iteration, warmup=False):
        options = self.train_options
        self.protector.train()
        self.gopt.zero_grad(set_to_none=True)
        utility, quantization = self.utility(iteration)
        exposure = utility.new_zeros(())
        if not warmup:
            risks = []
            with frozen_adversary(self.adversary):
                for _, episodes in self.sample_groups():
                    h, c, y, _, _ = encode_episodes(self.protector, self.store, episodes, iteration,
                                                    "training", self.seed, self.device)
                    confusion = confusion_loss(self.adversary(h, c)).mean()
                    marginal = marginal_linkage_gain(self.adversary, h, c, y, options["contexts"],
                                                       not options.get("signed_gain", False), self.context_rng)
                    risks.append(confusion + options["beta"] * marginal)
                exposure = exposure_group_risk(risks, options["tau"], options.get("mean_groups", False))
        loss = utility + options["lambda_q"] * quantization + options["lambda_e"] * exposure
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.protector.parameters(), 5)
        self.gopt.step()
        return {"loss": float(loss.detach()), "utility": float(utility.detach()),
                "quantization": float(quantization.detach()), "exposure": float(exposure.detach())}

    @torch.no_grad()
    def validate(self):
        self.protector.eval()
        state = self.rng.bit_generator.state
        self.rng = np.random.default_rng(self.seed + 701)
        try:
            utility, quantization = self.utility(0, "validation")
            risks = []
            with frozen_adversary(self.adversary):
                for family, length in self.groups:
                    if len(self.builder.eligible("validation", family, length, cohort_length=length)) < 2:
                        continue
                    plan = self.builder.build("validation", family, length, draws=1, cohort_length=length)
                    plan = plan.subset(np.arange(min(len(plan), 8)))
                    h, c, y, _, _ = encode_episodes(self.protector, self.store, plan, 0, "validation", self.seed, self.device)
                    marginal = marginal_linkage_gain(self.adversary, h, c, y, 1,
                                                       generator=torch.Generator().manual_seed(self.seed))
                    risks.append(confusion_loss(self.adversary(h, c)).mean() + self.train_options["beta"] * marginal)
            risk = exposure_group_risk(risks, self.train_options["tau"], self.train_options.get("mean_groups", False)) if risks else utility * 0
            return float(utility + self.train_options["lambda_q"] * quantization + self.train_options["lambda_e"] * risk)
        finally:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = state

    def save(self, name, step):
        torch.save({"schema_version": 1, "kind": "protector",
                    "protector": self.protector.state_dict(), "adversary": self.adversary.state_dict(),
                    "method": self.method, "model_options": self.model_options, "config": self.config,
                    "gopt": self.gopt.state_dict(), "aopt": self.aopt.state_dict(),
                    "step": step, "best_validation": self.best, "numpy_rng": self.rng.bit_generator.state,
                    "context_rng": self.context_rng.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "manifest_sha256": hashlib.sha256(self.store.path.read_bytes()).hexdigest()}, self.output / name)

    def resume(self, path):
        saved = read_checkpoint(path, self.device, "protector")
        if saved["manifest_sha256"] != hashlib.sha256(self.store.path.read_bytes()).hexdigest():
            raise ValueError("Cannot resume with a different feature manifest")
        self.protector.load_state_dict(saved["protector"])
        self.adversary.load_state_dict(saved["adversary"])
        self.gopt.load_state_dict(saved["gopt"])
        self.aopt.load_state_dict(saved["aopt"])
        self.start, self.best = saved["step"], saved["best_validation"]
        self.rng.bit_generator.state = saved["numpy_rng"]
        self.context_rng.set_state(saved["context_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])

    def run(self):
        options = self.train_options
        with (self.output / "training.jsonl").open("a", encoding="utf-8") as log:
            for iteration in range(self.start, options["steps"]):
                progress = iteration / max(1, options["steps"] - 1)
                for optimizer, initial in ((self.gopt, options["protector_lr"]), (self.aopt, options["attacker_lr"])):
                    optimizer.param_groups[0]["lr"] = options["min_lr"] + (initial - options["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2
                warmup = iteration < options["warmup_steps"]
                attack = 0.0
                if not warmup:
                    for substep in range(options["attacker_steps"]):
                        attack = self.attack_step(iteration, substep)
                metrics = self.protector_step(iteration, warmup)
                metrics.update(step=iteration + 1, attacker_loss=attack, warmup=warmup)
                if (iteration + 1) % options["validate_every"] == 0 or iteration + 1 == options["steps"]:
                    metrics["validation_loss"] = self.validate()
                    if metrics["validation_loss"] < self.best:
                        self.best = metrics["validation_loss"]
                        self.save("best.pt", iteration + 1)
                    self.save("last.pt", iteration + 1)
                    print(f"Protector {iteration + 1}/{options['steps']}, loss={metrics['loss']:.4f}, validation={metrics['validation_loss']:.4f}", flush=True)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
        model, _ = load_protector(self.output / "best.pt", self.device)
        return model
