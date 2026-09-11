"""Build the paired notebook, execute every cell, and keep inspectable plot outputs."""

import argparse
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "notebooks/01_forecasts_and_diagnostics.ipynb"
    )
    args = parser.parse_args()
    notebook = jupytext.read(root / "notebooks/01_forecasts_and_diagnostics.py")
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}
    )
    # Use the invoking environment, not an unrelated user-installed Python kernel.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client.execute()
    nbformat.validate(notebook)
    images = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )
    if images < 6:
        raise RuntimeError(f"Expected six plot outputs, got {images}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {code_cells} code cells; {images} PNG outputs; saved {args.output.name}")


if __name__ == "__main__":
    main()
