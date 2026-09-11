"""Existing linear point estimator -> fitted scales -> held-out intervals.

No downloads, GPU or private data. Numbers are synthetic checks, not paper results.
"""

import json
from pathlib import Path

import torch
from torch import nn

from sparc import SPARC, LinearLastLayerAdapter


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(162)
    weights = torch.randn(4, 2, generator=generator)

    def data(n):
        x = torch.randn(n, 4, generator=generator)
        noise = torch.randn(n, 2, generator=generator) * torch.tensor([0.3, 0.6])
        return x, (x @ weights + 0.4 + noise)[:, None, :]

    # This step represents the user's already trained point estimator.
    train_x, train_y = data(600)
    design = torch.cat([train_x, torch.ones(600, 1)], dim=1)
    coefficients = torch.linalg.lstsq(design, train_y[:, 0]).solution
    model = nn.Linear(4, 2).eval()
    with torch.no_grad():
        model.weight.copy_(coefficients[:-1].T)
        model.bias.copy_(coefficients[-1])

    fit_x, fit_y = data(600)
    cal_x, cal_y = data(600)
    test_x, test_y = data(2000)
    adapter = LinearLastLayerAdapter(model, model)
    uq = SPARC(adapter, model_id="linear-demo-v1/raw-input-v1", alpha=0.05)
    uq.fit([(fit_x, fit_y)])
    uq.calibrate([(cal_x, cal_y)])
    prediction = uq.predict(test_x)
    torch.testing.assert_close(prediction.mean[:, 0], model(test_x))
    coverage = ((test_y >= prediction.lower) & (test_y <= prediction.upper)).double().mean()

    Path("artifacts").mkdir(exist_ok=True)
    uq.save("artifacts/linear-uq.pt")
    restored = SPARC.load("artifacts/linear-uq.pt", adapter, model_id=uq.model_id)
    torch.testing.assert_close(prediction.lower, restored.predict(test_x).lower)
    print(
        json.dumps(
            {
                "data": "synthetic; not a paper benchmark",
                "mean_unchanged": True,
                "empirical_mean_marginal_coverage": coverage.item(),
                "mean_interval_width": (prediction.upper - prediction.lower).mean().item(),
                "kappa_min": prediction.kappa.min().item(),
                "n_fit": uq.leverage.n_fit,
                "n_calibration": uq.calibration.n_calibration,
                "reload_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
