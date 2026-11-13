"""Re-run the headline full-sample size/power experiment from public calibration parameters.

Example smoke run: python code/run_public_benchmark.py --reps 2 --bootstrap 49
For paper-scale Monte Carlo, use --reps 500 --bootstrap 200; this is computationally expensive.
"""

import argparse

import benchmark as bm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("jkv", "treasury_cmt"), default="jkv")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=49)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.0, help="AR(1) coefficient of simulated errors")
    parser.add_argument("--tau", type=int, choices=(160, 240, 320), default=240)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.rho < 1:
        parser.error("--rho must be between 0 (inclusive) and 1 (exclusive)")
    if args.reps < 1 or args.bootstrap < 1 or args.jobs < 1:
        parser.error("--reps, --bootstrap and --jobs must be positive")

    model, _ = bm.load_model(args.dataset)
    common = {"generator": "w9", "T": 480, "tau": args.tau,
              "dataset": args.dataset, "model": model, "resid_rho": args.rho}
    null = {**common, "setup": "null3"}
    change = {**common, "setup": "break", "strength": args.strength}
    test = bm.FullSample(d=3, B=args.bootstrap)
    seeds = bm.seeds("final", args.reps)
    null_out = bm.run_cell(test, null, seeds, jobs=args.jobs)
    change_out = bm.run_cell(test, change, seeds, jobs=args.jobs)
    size = bm.size(null_out, null, args.alpha)
    power = bm.power(change_out, change, args.alpha)
    print(f"{args.dataset}: N={model[3][0].shape[0]}, T=480, d=3, "
          f"tau={args.tau}, strength={args.strength}, rho={args.rho}")
    print(f"reps={args.reps}, bootstrap={args.bootstrap}, alpha={args.alpha}")
    print(f"size={size['rate']:.1%} (MC s.e. {size['se']:.1%})")
    print(f"power={power['rate']:.1%} (MC s.e. {power['se']:.1%})")


if __name__ == "__main__":
    main()
