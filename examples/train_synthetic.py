"""Train the public manufactured demo; never loads original CFD or checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from pemfc_public.model import DomainNetwork
from pemfc_public.synthetic import sample_plan, loss_chunks
from pemfc_public.training import take_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--physics", type=int, default=20)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("demo_outputs"))
    parser.add_argument("--resume", type=Path, help="Only a trusted checkpoint from this synthetic demo")
    args = parser.parse_args()
    if args.warmup < 0 or args.physics < 0 or args.warmup + args.physics == 0:
        parser.error("Use a positive total update count")
    if args.points < 2 or not 0 < args.lr < 1:
        parser.error("Invalid point count/learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = DomainNetwork().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    history, start = [], 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format"] != "public_synthetic_only_v1":
            raise ValueError("Not a public synthetic-demo checkpoint")
        expected = {key: getattr(args, key) for key in ("warmup", "physics", "points", "seed", "lr")}
        if saved["settings"] != expected:
            raise ValueError("Resume requires identical warmup/physics/points/seed/lr")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # Adam step tensors remain on CPU unless capturable=True; moments follow parameters.
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].to(args.device)
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["torch_cpu"])
        if args.device == "cuda" and saved["torch_cuda"]:
            torch.cuda.set_rng_state_all(saved["torch_cuda"])
        start, history = saved["completed"], saved["history"]
    settings = {key: getattr(args, key) for key in ("warmup", "physics", "points", "seed", "lr")}
    for step in range(start, args.warmup + args.physics):
        physics = step >= args.warmup
        alpha = (step - args.warmup + 1) / max(args.physics, 1) if physics else 0.
        plan = sample_plan(args.points, args.device, generator)
        record = take_step(model, optimizer, loss_chunks(model, plan, physics), alpha)
        record.update(step=step + 1, alpha=alpha, stage="physics" if physics else "data")
        history.append(record)
        report = {"scope": "manufactured diffusion; NOT PEMFC results",
                  "completed": step + 1, "torch_version": str(torch.__version__),
                  "device": args.device, "settings": settings, "history": history}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        checkpoint = dict(format="public_synthetic_only_v1", model=model.state_dict(),
                          optimizer=optimizer.state_dict(), generator=generator.get_state(),
                          torch_cpu=torch.get_rng_state(),
                          torch_cuda=torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                          settings=settings, completed=step + 1, history=history)
        pending = args.output / "checkpoint.pending"
        torch.save(checkpoint, pending)
        pending.replace(args.output / "last.pt")
        print(f"{step + 1}: stage={record['stage']}, alpha={alpha:.3g}, objective={record['objective']:.6g}", flush=True)
    print("Synthetic demo finished. This does not reproduce the PEMFC experiment.")


if __name__ == "__main__":
    main()
