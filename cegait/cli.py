"""Command-line entry points for CE-Gait training and evaluation."""
import argparse
import json
from pathlib import Path
from .config import load_config, resolve_path, seed_everything, device_for


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="CE-Gait: training, protected-template issuance, and security evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("checkpoint-info", help="Inspect checkpoint metadata and parameter compatibility")
    info.add_argument("--checkpoint", default="checkpoints/best.pt")
    prepare = sub.add_parser("prepare", help="Index preprocessed silhouette sequences")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--dataset", choices=["casia_b", "gait3d"], required=True)
    prepare.add_argument("--partition", help="Official Gait3D partition JSON")
    prepare.add_argument("--output", default="manifests/sequences.json")
    pretrain = sub.add_parser("pretrain", help="Pretrain IHGR-Net on development identities")
    pretrain.add_argument("--manifest", required=True)
    pretrain.add_argument("--output", default="outputs/ihgr/ihgr.pt")
    pretrain.add_argument("--steps", type=positive_integer, default=80000)
    pretrain.add_argument("--identities", type=positive_integer, default=8)
    pretrain.add_argument("--clips", type=positive_integer, default=4)
    pretrain.add_argument("--seed", type=int, default=42)
    pretrain.add_argument("--resume")
    pretrain.add_argument("--device", default="auto")
    extract = sub.add_parser("extract", help="Extract frozen IHGR-Net local and pooled descriptors")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--output", default="data/casia_b/manifest.json")
    extract.add_argument("--device", default="auto")
    for name in ("train", "audit", "experiment"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/casia_b.yaml")
        command.add_argument("--manifest", help="Feature manifest; overrides the configured path")
        command.add_argument("--output", default=f"outputs/{name}")
        command.add_argument("--device", help="cpu, cuda, or auto; overrides the configuration")
        command.add_argument("--steps", type=positive_integer)
        command.add_argument("--auditor-steps", type=positive_integer)
        command.add_argument("--method")
        command.add_argument("--seed", type=int)
        command.add_argument("--checkpoint")
        if name == "train":
            command.add_argument("--resume")
        elif name == "experiment":
            command.add_argument("--name", choices=[f"e{i}" for i in range(1, 8)], required=True)
        else:
            command.add_argument("--kind", choices=["query", "deepsets", "transformer"], default="query")
            command.add_argument("--knowledge", choices=["secret", "disclosed", "disclosure_control", "auxiliary", "auxiliary_control"], default="secret")
            command.add_argument("--length", type=positive_integer, default=4)
            command.add_argument("--family")
            command.add_argument("--candidate", default="nm-06@090")
    issue = sub.add_parser("issue", help="Issue a protected template from an enrolled silhouette sequence")
    issue.add_argument("--backbone", required=True)
    issue.add_argument("--checkpoint", required=True)
    issue.add_argument("--sequence", required=True)
    issue.add_argument("--key-file", required=True, help="Binary file containing a 16-byte domain key")
    issue.add_argument("--output", default="outputs/template.npy")
    issue.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    initial_seed = getattr(args, "seed", None)
    seed_everything(42 if initial_seed is None else initial_seed)
    device = device_for(getattr(args, "device", None) or "auto")
    if args.command == "checkpoint-info":
        from .checkpoint import checkpoint_summary
        print(json.dumps(checkpoint_summary(resolve_path(args.checkpoint)), indent=2))
        return
    elif args.command == "prepare":
        from .data import build_sequence_manifest
        result = build_sequence_manifest(args.root, args.dataset, resolve_path(args.output), args.partition)
    elif args.command == "pretrain":
        from .training.backbone_trainer import pretrain_backbone
        if args.identities < 2 or args.clips < 2:
            parser.error("Pretraining requires at least two identities and two clips per identity")
        result = pretrain_backbone(resolve_path(args.manifest), resolve_path(args.output), args.steps,
                                   device, seed=args.seed, identities_per_batch=args.identities,
                                   clips_per_identity=args.clips, resume=resolve_path(args.resume) if args.resume else None)
    elif args.command == "extract":
        from .training.backbone_trainer import extract_features
        result = extract_features(resolve_path(args.manifest), resolve_path(args.checkpoint), resolve_path(args.output), device)
    elif args.command == "issue":
        import numpy as np
        from .api import CEGaitPipeline
        from .data import load_sequence
        pipeline = CEGaitPipeline.from_checkpoints(resolve_path(args.backbone), resolve_path(args.checkpoint), device)
        images, _ = load_sequence(resolve_path(args.sequence))
        issued = pipeline.issue(images.unsqueeze(0), resolve_path(args.key_file).read_bytes())
        result = resolve_path(args.output)
        if result.suffix != ".npy":
            parser.error("Template output must have a .npy extension")
        result.parent.mkdir(parents=True, exist_ok=True)
        np.save(result, issued["payload"])
    else:
        from .data import FeatureStore, write_json
        config = load_config(args.config)
        if args.steps is not None:
            config["training"]["steps"] = args.steps
        if args.auditor_steps is not None:
            config["auditing"]["steps"] = args.auditor_steps
        if args.method:
            config["method"] = args.method
        if args.seed is not None:
            config["seed"] = args.seed
        device = device_for(args.device or config.get("device", "auto"))
        seed_everything(config.get("seed", 42), config.get("threads", 4))
        manifest = resolve_path(args.manifest or config["manifest"])
        if not manifest.is_file():
            parser.error(f"Feature manifest does not exist: {manifest}. Run prepare and extract first.")
        store = FeatureStore(manifest)
        if store.dataset != config["dataset"]:
            parser.error("The feature manifest dataset differs from the configuration")
        output = resolve_path(args.output)
        if args.command == "train":
            from .training.alternate_trainer import AlternateTrainer
            if args.checkpoint:
                parser.error("Use --resume to continue protector training")
            trainer = AlternateTrainer(store, config, output, device)
            if args.resume:
                trainer.resume(resolve_path(args.resume))
            trainer.run()
            result = output / "best.pt"
        elif args.command == "audit":
            from .baselines import make_protector
            from .training.alternate_trainer import load_protector
            from .experiments import audit_case
            selected_checkpoint = args.checkpoint or (config.get("checkpoint") if config.get("method", "cegait") == "cegait" else None)
            if selected_checkpoint:
                model, saved = load_protector(resolve_path(selected_checkpoint), device)
                if saved["config"]["dataset"] != store.dataset:
                    parser.error("The protector checkpoint belongs to another dataset")
            else:
                model = make_protector(config.get("method", "cegait"), **config.get("model", {})).to(device)
                if any(parameter.requires_grad for parameter in model.parameters()):
                    parser.error("An audit of a learned protector requires --checkpoint")
            lengths = sorted(set([1, 2, 4, args.length]))
            result = audit_case(model, store, config, output, device, args.family, lengths,
                                [args.length], args.candidate, args.kind, args.knowledge, args.length)
            write_json(output / "summary.json", result)
        else:
            from .experiments import run_experiment
            if args.checkpoint and args.name in {"e1", "e2", "e7"}:
                parser.error("E1, E2, and E7 train their own variants; --checkpoint applies to E3--E6")
            selected_checkpoint = args.checkpoint
            if selected_checkpoint is None and args.name in {"e3", "e4", "e5", "e6"} and config.get("method", "cegait") == "cegait":
                selected_checkpoint = config.get("checkpoint")
            result = run_experiment(args.name, store, config, output, device,
                                    resolve_path(selected_checkpoint) if selected_checkpoint else None)
    print(f"Saved: {result}" if isinstance(result, (Path, str)) else "Completed.")
