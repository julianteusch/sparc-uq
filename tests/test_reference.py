"""Optional numerical parity against the unchanged internal research source.

Only the three named pure tensor functions are imported via AST so the HPO and
dataset frameworks are not dependencies of this library. This is not an end-to-end
benchmark reproduction and must not be reported as one.
"""

import ast
import os
from pathlib import Path

import pytest
import torch

from sparc import FeatureLeverage, inflate_factor


def test_research_core_parity():
    root = os.environ.get("SPARC_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set SPARC_REFERENCE_ROOT to the internal simlpe-pie-mp checkout")
    path = Path(root) / "src/simple_mp/hpo/evaluators/hybrid.py"
    names = {"_kappa_from_lambda_inv", "_kappa_from_lambda_inv_timewise", "_scale_L_t_rows"}
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    generator = torch.Generator().manual_seed(42)
    train = torch.randn(100, 6, 9, generator=generator, dtype=torch.float64)
    test = torch.randn(12, 6, 9, generator=generator, dtype=torch.float64)
    leverage = FeatureLeverage(0.2).fit([train])
    inverse = torch.cholesky_inverse(leverage.cholesky)
    reference = namespace["_kappa_from_lambda_inv_timewise"](X=test, Lambda_inv_t=inverse)
    torch.testing.assert_close(leverage(test), reference)
    temporal = torch.eye(6, dtype=torch.float64).expand(12, -1, -1)
    scaled = namespace["_scale_L_t_rows"](temporal, sqrt_scale=reference.sqrt())
    torch.testing.assert_close(inflate_factor(temporal, leverage(test), channels=1), scaled)
