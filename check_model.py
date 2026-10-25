"""
Quick sanity check — run this FIRST, before touching any dataset.
Confirms the model builds, runs a forward pass, and lands close to the
base paper's reported 13.02M parameters / 50.89 MB.

Usage:
    python check_model.py
"""

import torch
from model import build_baseline_model, count_parameters, model_size_mb


def main():
    model = build_baseline_model(num_classes=38)
    model.eval()

    n_params = count_parameters(model)
    size_mb = model_size_mb(model)

    print(f"Trainable parameters : {n_params:,}  ({n_params / 1e6:.2f}M)")
    print(f"Approx. storage size : {size_mb:.2f} MB")
    print(f"Paper reports        : 13,020,000 params (~13.02M), 50.89 MB")
    print()

    # Forward pass with a dummy batch to confirm shapes are correct end-to-end
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape  : {tuple(dummy_input.shape)}")
    print(f"Output shape : {tuple(output.shape)}  (expected: (2, 38))")

    assert output.shape == (2, 38), "Output shape mismatch!"
    print("\nModel builds and runs correctly.")


if __name__ == "__main__":
    main()
