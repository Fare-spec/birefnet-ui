#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForImageSegmentation


DEFAULT_MODELS = [
    ("birefnet-lite", "ZhengPeng7/BiRefNet_lite", "birefnet-lite.ts"),
    ("birefnet-base", "ZhengPeng7/BiRefNet", "birefnet-base.ts"),
    ("birefnet-hr", "ZhengPeng7/BiRefNet_HR", "birefnet-hr.ts"),
]


@dataclass
class ExportSpec:
    model_id: str
    hf_repo: str
    filename: str


class BiRefNetTraceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        return output


def parse_model(value: str) -> ExportSpec:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("expected id|huggingface_repo|filename")
    return ExportSpec(parts[0], parts[1], parts[2])


def default_specs() -> list[ExportSpec]:
    return [ExportSpec(*item) for item in DEFAULT_MODELS]


def export_model(spec: ExportSpec, output_dir: Path, image_size: int, force: bool) -> None:
    output_path = output_dir / spec.filename
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"Skipping {spec.model_id}: {output_path} already exists")
        return

    print(f"Loading {spec.hf_repo}")
    model = AutoModelForImageSegmentation.from_pretrained(
        spec.hf_repo,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.cpu()
    model.eval()

    wrapper = BiRefNetTraceWrapper(model).cpu().eval()
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"Tracing {spec.model_id}")
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced.eval())

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"Writing {output_path}")
    traced.save(str(tmp_path))
    tmp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export official BiRefNet Hugging Face models to TorchScript."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/models"),
        help="Directory where .ts files are written.",
    )
    parser.add_argument(
        "--model",
        type=parse_model,
        action="append",
        help="Model to export as id|huggingface_repo|filename. Can be repeated.",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true", help="Overwrite existing .ts files.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = args.model or default_specs()

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    for spec in specs:
        export_model(spec, args.output_dir, args.image_size, args.force)


if __name__ == "__main__":
    main()
