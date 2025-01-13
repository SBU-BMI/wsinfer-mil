"""Command line interface for SpinPath."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from PIL import Image

from spinpath.client.hfmodel import load_torchscript_model_from_hf
from spinpath.client.localmodel import Model
from spinpath.client.localmodel import load_torchscript_model_from_filesystem
from spinpath.inference import infer_one_slide

logger = logging.getLogger(__name__)


def _load_tissue_mask(path: str | Path) -> Image.Image:
    logger.info(f"Loading tissue mask: {path}")
    with Image.open(path) as tissue_mask:
        tissue_mask.load()
    if tissue_mask.mode != "1":
        logger.info(
            f"Converting tissue mask from mode '{tissue_mask.mode}' to mode '1'"
        )
        tissue_mask = tissue_mask.convert("1")
    return tissue_mask


def _run_impl(
    wsi_path: Path,
    model: Model,
    tissue_mask_path: Path | None,
    num_workers: int,
    tablefmt: str,
    json: bool,
    quantize: bool,
) -> None:
    tissue_mask: Image.Image | None = None
    if tissue_mask_path is not None:
        tissue_mask = _load_tissue_mask(tissue_mask_path)

    result = infer_one_slide(
        slide_path=wsi_path,
        model=model,
        tissue_mask=tissue_mask,
        num_workers=num_workers,
        quantize=quantize,
    )

    if json:
        click.echo(result.to_json(fp=None))
    else:
        click.echo(result.to_str_table(tablefmt=tablefmt))


@click.group()
def cli() -> None:
    """Run specimen-level inference using pre-trained models."""


@cli.command()
@click.option(
    "-m", "--hf-repo-id", required=True, help="HuggingFace Hub repo ID of the model"
)
@click.option(
    "-i",
    "--wsi-path",
    required=True,
    help="Path to the whole slide image",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--hf-repo-revision", help="Revision of the HuggingFace Hub repo", default=None
)
@click.option(
    "--tissue-mask-path",
    help="Path to the tissue mask for this image",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "-j",
    "--num-workers",
    help="Number of workers to use during patch feature extraction. -1 (default) uses"
    " all cores.",
    type=click.IntRange(min=-1, max=os.cpu_count()),
    default=-1,
    show_default=True,
)
@click.option(
    "--table-format",
    help="Format of the output table with results. See Python Tabulate for a list of"
    " options.",
)
@click.option(
    "--json", is_flag=True, help="Print the model outputs (and attention) as JSON"
)
@click.option("--quantize", is_flag=True, help="Quantize embedding model to float16")
def run(
    *,
    hf_repo_id: str,
    wsi_path: Path,
    hf_repo_revision: str | None,
    tissue_mask_path: Path | None,
    num_workers: int,
    table_format: str,
    json: bool,
    quantize: bool,
) -> None:
    model = load_torchscript_model_from_hf(hf_repo_id, hf_repo_revision)
    if num_workers == -1:
        num_workers = os.cpu_count() or 0
    _run_impl(
        wsi_path=wsi_path,
        model=model,
        tissue_mask_path=tissue_mask_path,
        num_workers=num_workers,
        tablefmt=table_format,
        json=json,
        quantize=quantize,
    )


@cli.command()
@click.option("-m", "--model-path", help="Path to a TorchScript model")
@click.option("-c", "--model-config-path", help="Path to a JSON model config")
@click.option(
    "-i",
    "--wsi-path",
    help="Path to the whole slide image",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--tissue-mask-path",
    help="Path to the tissue mask for this image",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "-j",
    "--num-workers",
    help="Number of workers to use during patch feature extraction",
    type=click.IntRange(min=0, max=os.cpu_count()),
    default=4,
    show_default=True,
)
@click.option(
    "--table-format",
    help="Format of the output table with results. See Python Tabulate for a list of"
    " options.",
)
@click.option(
    "--json", is_flag=True, help="Print the model outputs (and attention) as JSON"
)
@click.option("--quantize", is_flag=True, help="Quantize embedding model to float16")
def runlocal(
    *,
    model_path: Path,
    model_config_path: Path,
    wsi_path: Path,
    tissue_mask_path: Path | None,
    num_workers: int,
    table_format: str,
    json: bool,
    quantize: bool,
) -> None:
    model = load_torchscript_model_from_filesystem(model_path, model_config_path)
    _run_impl(
        wsi_path=wsi_path,
        model=model,
        tissue_mask_path=tissue_mask_path,
        num_workers=num_workers,
        tablefmt=table_format,
        json=json,
        quantize=quantize,
    )
