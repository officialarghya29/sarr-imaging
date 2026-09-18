"""Registry of the SAR detection datasets this project supports.

Scope note
----------
There is no widely used public dataset named "SARR"; the project name is adopted
here as the *system* name, and the pipeline is deliberately dataset-agnostic so
the research questions can be tested across several public SAR benchmarks. The
development order is intentionally cheapest-first (see ``RECOMMENDED_ORDER``):
iterate on a small dataset until the pipeline and modules are trustworthy, then
scale to the large multi-class benchmark for the final numbers.

Every entry records its licence and citation because none of these datasets may
be redistributed by this repository — only the *code* is MIT licensed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DatasetSpec", "DATASETS", "RECOMMENDED_ORDER", "get_dataset", "list_datasets"]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one SAR detection dataset.

    Args:
        name: Registry key.
        title: Human-readable name.
        classes: Class names in the dataset's native order.
        source: Official landing page or mirror used to obtain the data.
        annotation_format: One of ``"voc"``, ``"coco"``, ``"yolo"``, ``"dota"``.
        license: Redistribution terms (this repo never redistributes the data).
        citation: BibTeX key / short citation for the dataset paper.
        approx_images: Approximate image count, for planning.
        recommended_imgsz: Input size that suits the dataset's object scale.
        tier: ``"pilot"`` datasets are small enough to iterate on quickly;
            ``"benchmark"`` datasets are for final reported numbers.
        notes: Practical notes (splits, quirks, leak risk).
        kaggle_slugs: Candidate ``kagglehub`` dataset slugs for automated fetch.
        hf_repos: Candidate Hugging Face dataset repos for automated fetch.
    """

    name: str
    title: str
    classes: tuple[str, ...]
    source: str
    annotation_format: str
    license: str
    citation: str
    approx_images: int
    recommended_imgsz: int
    tier: str
    notes: str = ""
    kaggle_slugs: tuple[str, ...] = field(default_factory=tuple)
    hf_repos: tuple[str, ...] = field(default_factory=tuple)

    @property
    def nc(self) -> int:
        return len(self.classes)


DATASETS: dict[str, DatasetSpec] = {
    "ssdd": DatasetSpec(
        name="ssdd",
        title="SSDD — SAR Ship Detection Dataset",
        classes=("ship",),
        source="https://github.com/TianwenZhang0825/Official-SSDD",
        annotation_format="voc",
        license="Research use; see the official repository",
        citation="Zhang et al., Multi-Scale Context Aggregation for SAR Ship Detection (2019)",
        approx_images=1160,
        recommended_imgsz=512,
        tier="pilot",
        notes=(
            "1160 images, 2456 ships, single class, PASCAL-VOC XML annotations. "
            "The official repo ships several split files (e.g. 'ImageSets/Main/train.txt'); "
            "use one of them rather than inventing a split. Small enough to run a full "
            "ablation grid end to end on a free Colab GPU."
        ),
        kaggle_slugs=("gogogo1999/ssdd", "punitsharma/sar-ship-detection-ssdd", "ssdd/sar-ship-detection-dataset"),
    ),
    "hrsid": DatasetSpec(
        name="hrsid",
        title="HRSID — High-Resolution SAR Images Dataset",
        classes=("ship",),
        source="https://github.com/chaozhong2010/HRSID",
        annotation_format="coco",
        license="Research use; see the official repository",
        citation="Wei et al., HRSID: A High-Resolution SAR Images Dataset for Ship Detection (2020)",
        approx_images=5604,
        recommended_imgsz=800,
        tier="pilot",
        notes=(
            "5604 images from 99 Sentinel-1/2/TerraSAR-X scenes with COCO annotations. "
            "IMPORTANT: images are cropped from a small number of large scenes, so random "
            "splitting leaks near-identical patches across train/test. Prefer splitting by "
            "scene, and always run the leakage check in saryolo.data.splits."
        ),
        kaggle_slugs=("sovitrath/sar-ship-detection", "zhangyunsheng/hrsid-sar-ship-dataset"),
        hf_repos=("lgrzybowski/hrsid",),
    ),
    "sardet100k": DatasetSpec(
        name="sardet100k",
        title="SARDet-100K",
        classes=("ship", "aircraft", "car", "tank", "bridge", "harbor"),
        source="https://github.com/zcablii/SARDet_100K",
        annotation_format="coco",
        license="Research use; see the official repository",
        citation="Li et al., SARDet-100K: Towards Open-Source Benchmark and Toolkit for Large-Scale SAR Object Detection (NeurIPS 2024)",
        approx_images=116598,
        recommended_imgsz=800,
        tier="benchmark",
        notes=(
            "The current large-scale SAR detection benchmark: ~116k images and "
            "~245k instances across 6 classes, unified from 10 source datasets. This is the "
            "target for the paper's headline table. It is far too large for a free Colab "
            "session: budget for multi-session training with checkpoint resume, or Colab Pro."
        ),
        hf_repos=("benjamin-paine/sardet-100k", "ywyiing/SARDet-100K", "likyoo/SARDet-100K"),
    ),
    "sar_ship": DatasetSpec(
        name="sar_ship",
        title="SAR-Ship-Dataset",
        classes=("ship",),
        source="https://github.com/CAESAR-Radi/SAR-Ship-Dataset",
        annotation_format="dota",
        license="Research use; see the official repository",
        citation="Wang et al., SAR-Ship-Dataset: A Large-Scale SAR Ship Dataset (2019)",
        approx_images=43819,
        recommended_imgsz=512,
        tier="benchmark",
        notes=(
            "~43k chips from Sentinel-1 and Gaofen-3, annotated in DOTA-style rotated "
            "oriented boxes. Useful for the oriented-detection (Component 6) study: the "
            "annotations support meaningfully evaluating whether orientation helps."
        ),
    ),
}

#: Cheapest-first order: prove the pipeline on a small dataset, then scale up.
#: Ordered by approximate image count so each step costs more than the last;
#: `tests/test_data.py::test_recommended_order_is_cheapest_first` enforces this.
RECOMMENDED_ORDER: tuple[str, ...] = ("ssdd", "hrsid", "sar_ship", "sardet100k")


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset by registry key."""
    key = name.lower().replace("-", "_")
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {', '.join(sorted(DATASETS))}")
    return DATASETS[key]


def list_datasets() -> list[DatasetSpec]:
    """All registered datasets, in recommended development order."""
    return [DATASETS[k] for k in RECOMMENDED_ORDER if k in DATASETS]
