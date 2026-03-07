# List module key/values in a checkpointed torch module

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    args = ap.parse_args()

    m = torch.jit.load(args.infile, map_location="cpu")  # TorchScript
    sd = m.state_dict()

    print("num tensors:", len(sd))
    for i, k in enumerate(sd.keys()):
        print(" ", k)

    print("\nmodules:")
    for i, (name, mod) in enumerate(m.named_modules()):
        print(" ", name)


if __name__ == "__main__":
    main()
