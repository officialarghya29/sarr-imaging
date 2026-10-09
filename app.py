"""SARVO — Streamlit demo for SAR ship detection.

Run locally:

    streamlit run app.py

The interface is a thin layer over :mod:`saryolo.inference`: it never re-implements the model,
the preprocessing, or the box rendering. That is deliberate -- a demo that built its own
inference could display boxes the model never produced, and no test would see it. Everything
numeric or geometric shown here comes from the pipeline function that the test suite exercises.

Weights are **not** committed (they are a training artefact and the repository keeps them out of
git). Point the sidebar at a checkpoint, or set ``SARVO_WEIGHTS``. With no checkpoint the app
says so and offers no detection, rather than showing predictions from an untrained model.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from saryolo.inference import (
    CheckpointError,
    InferenceError,
    InvalidImageError,
    annotate_image,
    class_names,
    describe_model,
    load_detector,
    predict_image,
    resolve_weights,
    validate_and_decode,
)

#: Resolved from `SARVO_WEIGHTS`, then the documented default locations. `""` means nothing was
#: configured -- which the app reports as such rather than as a missing file at path `.`.
DEFAULT_WEIGHTS = resolve_weights() or ""
#: A local copy of the pilot's val chips, when present, lets the demo run without an upload.
LOCAL_SAMPLE_DIR = Path("datasets/processed/hrsid_real/images/val")

@st.cache_resource(show_spinner=False)
def _load_cached(weights: str, mtime: float):
    """Load a checkpoint once per (path, mtime) and reuse it for every request.

    Caching on the file's modification time means re-training the arm and re-running the app
    picks up the new weights instead of silently serving the old ones -- a cache keyed on the
    path alone would do exactly that.
    """
    return load_detector(weights, device="cpu")


def _weights_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def _load_or_error(weights: str):
    """Return (model, error_message). The error path never yields a model."""
    if not weights:
        return None, (
            "No checkpoint is configured. Set `SARVO_WEIGHTS` to your checkpoint path, or type "
            "one in the sidebar. The demo will not run predictions from an untrained model."
        )
    path = Path(weights)
    if not path.is_file():
        return None, (
            f"No checkpoint at `{path}`. Train an arm (`python -m saryolo train --exp "
            "configs/exp/SSAC-001_hrsid_ssac.yaml`) or point the sidebar at an existing "
            "checkpoint. The demo will not run predictions from an untrained model."
        )
    try:
        return _load_cached(weights, _weights_mtime(path)), None
    except CheckpointError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user, never swallowed silently
        return None, f"The checkpoint could not be loaded: {exc}"


def _sidebar() -> dict:
    st.sidebar.header("Model & inference")
    weights = st.sidebar.text_input("Checkpoint path", value=DEFAULT_WEIGHTS)
    imgsz = st.sidebar.select_slider("Input size (px)", options=[320, 512, 640, 800], value=640)
    conf = st.sidebar.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01)
    iou = st.sidebar.slider("NMS IoU", 0.1, 0.9, 0.7, 0.05)
    st.sidebar.caption(
        "The confidence default (0.25) is a *deployment* threshold. The published metrics use "
        "0.001, which traces the full precision/recall curve — the two are not interchangeable."
    )
    return {"weights": weights, "imgsz": imgsz, "conf": conf, "iou": iou}


def _render_intro() -> None:
    st.title("SARVO — SAR Acquisition-Robust ship detection")
    st.markdown(
        "**SARVO** (*SAR Acquisition-Robust Visual Optimization*) is a research prototype for "
        "object detection in synthetic-aperture radar imagery. Its core mechanism is "
        "**Scatter-Selective Adaptive Computation (SSAC)**: a cheap shared path runs everywhere, "
        "a multi-scale SAR statistic (log-ratio + local coefficient of variation) drives a "
        "per-region allocation, and expensive feature processing is added only where the "
        "evidence says it is useful."
    )
    with st.expander("Research status and honest scope", expanded=False):
        st.markdown(
            "- **This is a pilot, not a benchmark.** The model shown here was trained on a "
            "**200/60/60 image subset of HRSID** at 320 px, 40 epochs, CPU only, one class "
            "(ship). Published mAP50:95 is **0.30** — a real, measured, deliberately modest "
            "number, not a state-of-the-art claim.\n"
            "- **The accuracy claim for the adaptive allocation was withdrawn** after a "
            "three-seed check: the gain over a parameter-identical fixed-computation control "
            "did not reproduce. That negative is reported, not hidden.\n"
            "- **The efficiency result is scale-bound.** Sparse execution is not faster than "
            "dense at the 320 px training scale, but is faster (−8.7 to −13.5 %) at 640 px, "
            "where every governed level has more than one tile.\n"
            "- Full evidence: `paper/RESULTS.md`, `docs/claim_evidence_audit.md`, "
            "`reports/release_readiness.md`."
        )


def _limitations() -> None:
    with st.expander("Limitations", expanded=False):
        st.markdown(
            "- **One dataset, one subset, one class.** HRSID, 200/60/60, ship only. It is not "
            "validated on other sensors, resolutions, or target types.\n"
            "- **No cross-sensor or cross-resolution result.** The acquisition-conditioning "
            "machinery exists and is tested, but the generalisation claim is unmeasured.\n"
            "- **CPU-only here.** GPU execution paths are written and guarded but have not been "
            "executed on this host.\n"
            "- **Detections are the model's raw output** at the chosen confidence threshold. "
            "Low-confidence boxes are common on this pilot; that is the model, not the display.\n"
            "- This demo performs **research inference**, not operational surveillance."
        )


def _pick_sample():
    """Offer a local val chip when one exists; HRSID chips are not redistributed with the code."""
    if not LOCAL_SAMPLE_DIR.is_dir():
        return None
    chips = sorted(LOCAL_SAMPLE_DIR.glob("*.jpg"))
    if not chips:
        return None
    names = ["— none —"] + [p.name for p in chips[:25]]
    choice = st.selectbox("Or use a local sample chip (HRSID, not committed)", names, index=0)
    return None if choice == "— none —" else LOCAL_SAMPLE_DIR / choice


def main() -> None:
    st.set_page_config(page_title="SARVO — SAR ship detection", page_icon="🛰", layout="wide")
    cfg = _sidebar()
    _render_intro()

    model, load_error = _load_or_error(cfg["weights"])
    if load_error:
        st.error(load_error)
        st.info(
            "**Recovery:** check the path in the sidebar, or set `SARVO_WEIGHTS` to your "
            "checkpoint. The repository intentionally does not commit trained weights."
        )
    else:
        info = describe_model(model, cfg["weights"])
        cols = st.columns(4)
        cols[0].metric("Parameters", f"{info['params_M']:.2f} M")
        cols[1].metric("Checkpoint", f"{info['checkpoint_MB']} MB" if info["checkpoint_MB"] else "—")
        cols[2].metric("Device", info["device"])
        cols[3].metric("Classes", ", ".join(info["classes"].values()))

    st.divider()
    st.subheader("Run a detection")
    uploaded = st.file_uploader(
        "Upload a SAR image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Grayscale SAR chips work best. Max 8 MB, 32–6000 px per side.",
    )
    sample = _pick_sample()

    raw: bytes | None = None
    filename = "upload"
    if uploaded is not None:
        raw, filename = uploaded.getvalue(), uploaded.name
    elif sample is not None:
        raw, filename = sample.read_bytes(), sample.name

    if raw is None:
        st.caption("Upload an image, or pick a local sample chip if one is available.")
        _limitations()
        return

    try:
        image = validate_and_decode(raw, filename)
    except InvalidImageError as exc:
        st.error(f"That file cannot be used: {exc}")
        st.info(
            "**Recovery:** re-export the image as PNG/JPG under 8 MB and between 32 and "
            "6000 px per side."
        )
        _limitations()
        return

    left, right = st.columns(2)
    rgb = image.array[:, :, ::-1]
    left.image(rgb, caption=f"Input — {image.width}×{image.height} px", width="stretch")

    if model is None:
        right.info("Detection is disabled until a checkpoint loads (see the message above).")
        _limitations()
        return

    if not st.button("Run Detection", type="primary"):
        right.caption("Press **Run Detection** to run the model on this image.")
        _limitations()
        return

    with st.spinner("Running SARVO inference …"):
        try:
            result = predict_image(
                model, image.array, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                device="cpu", weights=cfg["weights"],
            )
        except InferenceError as exc:
            st.error(f"Inference failed: {exc}")
            _limitations()
            return

    annotated = annotate_image(image.array, result.detections, class_names(model))
    right.image(
        annotated[:, :, ::-1],
        caption=f"{result.n_detections} detection(s) at conf ≥ {cfg['conf']:.2f}",
        width="stretch",
    )

    m = st.columns(3)
    m[0].metric("Detections", result.n_detections)
    m[1].metric("Latency", f"{result.latency_ms:.0f} ms")
    m[2].metric("Throughput", f"{1000 / max(result.latency_ms, 1e-6):.1f} img/s")
    st.caption(
        f"Measured on CPU at {cfg['imgsz']} px, including preprocessing and NMS. Latency is "
        "dominated by the first call of a session; repeated calls are faster."
    )

    if result.n_detections:
        st.dataframe(result.as_rows(), width="stretch", hide_index=True)
    else:
        st.warning(
            "No detections above the confidence threshold. This is a valid result for a pilot "
            "model — lower the threshold in the sidebar to see low-confidence boxes."
        )

    _limitations()


# Guarded so a test can import this module (and call its pure helpers) without launching the
# UI. Streamlit runs the main script with ``__name__ == "__main__"``, so ``streamlit run app.py``
# still executes ``main()``.
if __name__ == "__main__":
    main()
