from __future__ import annotations

import asyncio

from safebench.config import load_config
from safebench.evaluation.runner import run_benchmark
from safebench.plots.generate import generate_plots


async def main() -> None:
    config = load_config("config/benchmark.demo.yaml")
    artifacts = await run_benchmark(config)
    generate_plots(artifacts["records_csv"], config.output_dir / "plots")
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())

