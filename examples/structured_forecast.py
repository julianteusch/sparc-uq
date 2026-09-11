"""Custom adapter: constant-velocity means and graph-temporal covariance.

The covariance is specified here, not learned. This is an interface demonstration,
not a reproduction of SPARC's learned covariance head or its reported benchmarks.
"""

import json

import torch

from sparc import SPARC, Forecast, graph_temporal_factor


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(42)
    h, c = 6, 4  # Six horizons, two joints with two coordinates each.
    time = torch.linspace(0.1, 0.6, h)
    temporal = torch.linalg.cholesky(
        0.04 * 0.6 ** (torch.arange(h)[:, None] - torch.arange(h)).abs()
    )
    factor = graph_temporal_factor(
        temporal[None],
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        tau=0.5,
        epsilon=1.0,
        coordinates=2,
    )

    def existing_forecaster(x):
        return x[:, None, :c] + time[None, :, None] * x[:, None, c:]

    def adapter(x):
        mean = existing_forecaster(x)
        features = torch.cat(
            [
                x[:, None, :c].expand(-1, h, -1),
                time[None, :, None] * x[:, None, c:],
                torch.ones(x.shape[0], h, 1),
            ],
            dim=-1,
        )
        return Forecast(mean, features, covariance_factor=factor.expand(x.shape[0], -1, -1))

    def data(n):
        x = torch.randn(n, 2 * c, generator=generator)
        noise = torch.randn(n, h * c, generator=generator) @ factor[0].T
        return x, existing_forecaster(x) + noise.reshape(n, h, c)

    fit, calibration, test = data(400), data(400), data(1000)
    uq = SPARC(adapter, model_id="constant-velocity-v1/graph-v1")
    uq.fit([fit]).calibrate([calibration])
    p = uq.predict(test[0])
    torch.testing.assert_close(p.mean, existing_forecaster(test[0]))
    print(
        json.dumps(
            {
                "data": "synthetic; manually specified covariance",
                "mean_unchanged": True,
                "factor_shape": list(p.covariance_factor.shape),
                "empirical_mean_marginal_coverage": ((test[1] >= p.lower) & (test[1] <= p.upper))
                .double()
                .mean()
                .item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
