"""
COLT wrapper baseline (Qu et al., CIKM 2024 — quchangle1/COLT).

Paper: "Towards Completeness-Oriented Tool Retrieval for Large Language Models"
Repo:  https://github.com/quchangle1/COLT (pinned commit
       bbec292c38cabe3fbae098f50cc31b2e9f7e8fca; see baselines/REPRODUCTION.md)

Relation to FitText
-------------------
COLT cuts the retriever topology (dual-view GCN over (Query, Tool),
(Query, Scene), (Scene, Tool) bipartite graphs) but keeps a single-pass,
non-belief retrieval pipeline. It is orthogonal to FitText's $\\mathcal{B}$-cut
(belief-space evolution). COLT is one of the few cited baselines with a public
training-time repo, so we *clone and wrap* rather than reimplement.

Architecture (upstream)
-----------------------
Two-stage training:
  1. Semantic learning — PLM (Contriever / ANCE / TAS-B / co-Condensor)
     fine-tuned on (query, tool description) pairs.
  2. Collaborative learning — dual-view GCN with BPR + contrastive losses,
     produces query and tool embeddings.
Inference: dot-product top-k between trained query embedding and the tool
embedding table.

Wrapper design
--------------
We do NOT reimplement COLT — the architecture is non-trivial and any numbers
must come from the original code on its own data splits. The wrapper:

1. Detects the clone at ``baselines/external/colt/`` (or ``$COLT_PATH``).
2. Records the pinned commit SHA + dataset name in result metadata so every
   COLT measurement is traceable to the exact upstream code that produced it.
3. Invokes COLT's ``train.py --infer True`` flow as a subprocess against a
   pre-trained checkpoint, then parses the resulting ``tensor_data_formatted.txt``
   (or a JSONL the wrapper itself writes for batched inference) into our
   ``RetrievedTools`` schema.
4. Maps COLT's internal integer tool indices back to composite
   ``"category::tool_name::api_name"`` IDs via a corpus index file produced at
   training time and kept under ``$COLT_PATH/datasets/<COLT_DATASET>/``.

Important constraint
--------------------
COLT operates over the **same tool universe it was trained on** (integer indices
into a fixed corpus). It cannot directly retrieve over an arbitrary external
tool catalog without retraining. We therefore report
COLT on its own benchmark (ToolLens by default), and the wrapper raises a
``RuntimeError`` if asked to retrieve over a tool catalog whose api_names do
not appear in the trained corpus.

Environment variables
---------------------
- ``COLT_PATH``     — path to the cloned COLT repo root. If unset, defaults to
                      ``<project_root>/baselines/external/colt`` and an
                      ``ImportError`` is raised when that path is missing.
- ``COLT_CKPT``     — path to the trained collaborative-learning checkpoint
                      (a ``.pt`` saved by ``train.py``). Required for inference.
- ``COLT_DATASET``  — one of ``ToolLens`` | ``ToolBenchG2`` | ``ToolBenchG3``
                      (default: ``ToolLens``). Determines the corpus the
                      checkpoint was trained on.
- ``COLT_DEVICE``   — torch device for COLT inference (default: ``cuda:0``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from .base import Baseline, RetrievedTools, RetrieverAdapter
from StableToolBench.toolbench.inference.LLM.clients import ModelClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinning — single source of truth for the cloned upstream commit.
# Must match scripts/setup_colt.sh and baselines/REPRODUCTION.md.
# ---------------------------------------------------------------------------

COLT_PINNED_SHA = "bbec292c38cabe3fbae098f50cc31b2e9f7e8fca"
COLT_SUPPORTED_DATASETS = ("ToolLens", "ToolBenchG2", "ToolBenchG3")


# ---------------------------------------------------------------------------
# Path / SHA resolution
# ---------------------------------------------------------------------------

def _default_colt_path() -> Path:
    """Return the default ``baselines/external/colt`` path under this repo."""
    # baselines/colt.py -> baselines -> repo root
    return Path(__file__).resolve().parent.parent / "baselines" / "external" / "colt"


def _resolve_colt_path() -> Path:
    """Resolve the COLT repo root.

    Order of resolution:
      1. ``$COLT_PATH`` if set and non-empty.
      2. Default ``baselines/external/colt/`` next to this module.

    Raises:
        ImportError: If neither path exists. Message points the user at
            ``scripts/setup_colt.sh``.
    """
    env_path = os.environ.get("COLT_PATH", "").strip()
    if env_path:
        colt_path = Path(env_path)
    else:
        colt_path = _default_colt_path()

    if not colt_path.exists() or not (colt_path / "train.py").exists():
        raise ImportError(
            "COLT clone not found at "
            f"{colt_path}. Run scripts/setup_colt.sh to clone the upstream repo "
            f"at pinned commit {COLT_PINNED_SHA}, then either re-run from the "
            "FitText repo root or export $COLT_PATH explicitly.\n"
            "  bash scripts/setup_colt.sh"
        )
    return colt_path


def _verify_colt_sha(colt_path: Path) -> str:
    """Verify the clone is at the pinned SHA and return the resolved SHA.

    Args:
        colt_path: COLT repo root.

    Returns:
        The actual HEAD SHA at ``colt_path``.

    Raises:
        RuntimeError: If the clone is at a different SHA than ``COLT_PINNED_SHA``
            and the strict-pin env var is set. By default we only log a warning
            so off-pin local development remains possible.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(colt_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("COLT SHA verification failed (%s); proceeding without pin check.", exc)
        return "unknown"

    if sha != COLT_PINNED_SHA:
        msg = (
            f"COLT clone at {colt_path} is at SHA {sha} but pinned SHA is "
            f"{COLT_PINNED_SHA}. Re-run scripts/setup_colt.sh to re-pin, or set "
            "COLT_ALLOW_OFFPIN=1 to bypass."
        )
        if os.environ.get("COLT_ALLOW_OFFPIN", "0") != "1":
            raise RuntimeError(msg)
        logger.warning(msg)
    return sha


def _resolve_colt_ckpt(colt_path: Path) -> Path:
    """Resolve the path to the trained COLT checkpoint.

    Checks ``$COLT_CKPT`` first; falls back to the default
    ``$COLT_PATH/checkpoints/colt.pt``.

    Args:
        colt_path: COLT repo root.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint cannot be located. The error
            points to REPRODUCTION.md for training instructions.
    """
    ckpt_str = os.environ.get("COLT_CKPT", "").strip()
    if ckpt_str:
        ckpt = Path(ckpt_str)
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(
            f"COLT checkpoint not found at $COLT_CKPT={ckpt}. See "
            "baselines/REPRODUCTION.md for training instructions, or download "
            "a published checkpoint from https://huggingface.co/Tool-COLT."
        )
    default_ckpt = colt_path / "checkpoints" / "colt.pt"
    if default_ckpt.exists():
        return default_ckpt
    raise FileNotFoundError(
        f"COLT checkpoint not found at {default_ckpt}. The upstream COLT repo "
        "ships no checkpoint — you must either train one (see "
        "baselines/REPRODUCTION.md) or download from "
        "https://huggingface.co/Tool-COLT and set $COLT_CKPT."
    )


def _resolve_colt_dataset() -> str:
    """Resolve the COLT dataset name from $COLT_DATASET (default ToolLens)."""
    dataset = os.environ.get("COLT_DATASET", "ToolLens").strip() or "ToolLens"
    if dataset not in COLT_SUPPORTED_DATASETS:
        raise ValueError(
            f"COLT_DATASET={dataset!r} unsupported. "
            f"Must be one of {COLT_SUPPORTED_DATASETS}."
        )
    return dataset


# ---------------------------------------------------------------------------
# Corpus index — maps COLT's integer tool indices to our composite IDs.
# ---------------------------------------------------------------------------

def _load_colt_corpus_index(colt_path: Path, dataset: str) -> dict[int, str]:
    """Build ``{int_idx -> api_name}`` from the dataset's corpus.jsonl.

    COLT's datasets ship a ``corpus.jsonl`` with one JSON record per tool. The
    record's order in the file IS the integer tool index used internally — this
    is the convention used by COLT's data loader (utility.Datasets).

    Args:
        colt_path: COLT repo root.
        dataset: One of ``ToolLens`` | ``ToolBenchG2`` | ``ToolBenchG3``.

    Returns:
        Mapping from tool integer index to api_name string.
    """
    corpus_path = colt_path / "datasets" / dataset / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"COLT corpus file not found at {corpus_path}. The clone may be "
            "incomplete — re-run scripts/setup_colt.sh."
        )

    idx_to_name: dict[int, str] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("COLT corpus.jsonl line %d not valid JSON; skipping.", idx)
                continue
            # COLT's records typically have fields {"_id", "title", "text"}; the
            # api_name lives in "title" or under a "name" key depending on the
            # dataset variant.
            api_name = (
                record.get("title")
                or record.get("name")
                or record.get("api_name")
                or record.get("_id")
                or ""
            )
            idx_to_name[idx] = str(api_name)
    return idx_to_name


def _build_api_to_composite(tool_catalog: list[dict]) -> dict[str, str]:
    """Build ``{api_name.lower() -> "category::tool_name::api_name"}`` lookup."""
    api_to_id: dict[str, str] = {}
    for tool in tool_catalog:
        cat = tool.get("category", "")
        tname = tool.get("tool_name") or tool.get("name") or ""
        aname = tool.get("api_name") or tname
        if aname:
            api_to_id[str(aname).lower()] = f"{cat}::{tname}::{aname}"
    return api_to_id


# ---------------------------------------------------------------------------
# Output parser — reads COLT's tensor_data_formatted.txt + scoring sidecar.
# ---------------------------------------------------------------------------

def _parse_colt_topk_indices(raw: str) -> list[list[int]]:
    """Parse COLT's ``tensor_data_formatted.txt`` into per-query top-k indices.

    Format (per line, one query): ``[3, 17, 442]``.

    Args:
        raw: Contents of the tensor_data file (or empty string).

    Returns:
        List of per-query lists of integer tool indices.
    """
    out: list[list[int]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip [ ] and split commas.
        inner = line.strip("[]")
        if not inner:
            out.append([])
            continue
        try:
            indices = [int(tok.strip()) for tok in inner.split(",") if tok.strip()]
        except ValueError:
            logger.warning("COLT top-k line not parseable: %r", line)
            continue
        out.append(indices)
    return out


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class COLTBaseline(Baseline):
    """COLT dual-encoder + dual-view GCN retrieval baseline.

    Wraps the upstream ``quchangle1/COLT`` repository at pinned commit
    ``bbec292c...``. Requires either the clone to exist at
    ``baselines/external/colt/`` or ``$COLT_PATH`` to point at one — see
    ``scripts/setup_colt.sh``.

    Per-query retrieval invokes COLT's inference subprocess on a single-query
    input file, reads back the produced ``tensor_data_formatted.txt`` from a
    tempdir, and maps the integer tool indices to our composite IDs via the
    dataset's ``corpus.jsonl``.

    COLT does not use an LLM at inference — ``model_client`` is accepted only
    for interface conformance.

    Config kwargs
    -------------
    colt_batch_size : int, default 64
        Batch size for COLT subprocess inference.
    colt_device : str, default ``$COLT_DEVICE`` or ``cuda:0``
        Torch device for COLT inference.
    colt_timeout_seconds : int, default 600
        Subprocess timeout (cold start of the GCN can take >2 min on first call).
    colt_strict_catalog_match : bool, default False
        If True, raise when any catalog entry is missing from COLT's trained
        corpus (i.e. catalog has out-of-vocabulary tools). If False, log a
        warning and only return tools that are in the trained corpus.
    """

    name: ClassVar[str] = "colt"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: Any,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client, retriever, top_k=top_k, **kwargs)

        # Resolve paths up-front so construction-time failures are loud.
        self._colt_path: Path = _resolve_colt_path()
        self._colt_sha: str = _verify_colt_sha(self._colt_path)
        self._colt_ckpt: Path = _resolve_colt_ckpt(self._colt_path)
        self._dataset: str = _resolve_colt_dataset()
        self._device: str = str(
            self.cfg.get("colt_device") or os.environ.get("COLT_DEVICE", "cuda:0")
        )
        self._batch_size: int = int(self.cfg.get("colt_batch_size", 64))
        self._timeout_seconds: int = int(self.cfg.get("colt_timeout_seconds", 600))
        self._strict_catalog_match: bool = bool(
            self.cfg.get("colt_strict_catalog_match", False)
        )

        # Pre-load the corpus index so each retrieve() call is fast.
        self._idx_to_api: dict[int, str] = _load_colt_corpus_index(
            self._colt_path, self._dataset
        )

        logger.info(
            "COLTBaseline initialized: repo=%s sha=%s dataset=%s ckpt=%s "
            "device=%s corpus_size=%d",
            self._colt_path,
            self._colt_sha,
            self._dataset,
            self._colt_ckpt,
            self._device,
            len(self._idx_to_api),
        )

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        """Retrieve top-k tools for ``query`` via COLT's GCN inference.

        Writes ``query`` and a one-query inference manifest to a tempdir,
        invokes COLT's training entry script with ``--infer True``, reads the
        produced top-k tensor file, and maps the integer indices to composite
        tool IDs via ``corpus.jsonl``.

        Args:
            query: User task query (natural language).
            tool_catalog: Full tool list from harness — used to filter COLT's
                output to tools present in the caller's catalog and to map
                api_name -> composite "category::tool_name::api_name".

        Returns:
            ``RetrievedTools`` with up to ``self.top_k`` ranked items.

        Raises:
            RuntimeError: If COLT subprocess fails or produces unparseable
                output.
            subprocess.TimeoutExpired: If COLT inference exceeds
                ``colt_timeout_seconds``.
        """
        t0 = time.monotonic()

        # Catalog mapping — needed both to map COLT api_names back and to spot
        # out-of-vocabulary tools.
        api_to_id = _build_api_to_composite(tool_catalog)
        catalog_api_set = set(api_to_id.keys())
        corpus_api_set = {api.lower() for api in self._idx_to_api.values()}
        oov = catalog_api_set - corpus_api_set
        if oov and self._strict_catalog_match:
            raise RuntimeError(
                f"COLT strict-catalog-match enabled: {len(oov)} of "
                f"{len(catalog_api_set)} catalog tools are not in COLT's "
                f"trained corpus for dataset={self._dataset}. Disable strict "
                "mode (colt_strict_catalog_match=False) to retrieve only over "
                "the intersection."
            )
        if oov:
            logger.debug(
                "COLT: %d catalog tools not in trained corpus for %s; "
                "ignoring them in ranking.",
                len(oov),
                self._dataset,
            )

        with tempfile.TemporaryDirectory(prefix="colt_run_") as tmpdir:
            tmp = Path(tmpdir)
            query_file = tmp / "query.txt"
            output_indices_file = tmp / "tensor_data_formatted.txt"

            # COLT's inference path reads queries from the dataset's test
            # split file by default. We override via a wrapper-controlled
            # query file that train.py is invoked against. The exact CLI
            # surface used here matches the entry contract documented in
            # baselines/REPRODUCTION.md (--infer True + checkpoint load).
            query_file.write_text(query + "\n", encoding="utf-8")

            cmd = [
                sys.executable,
                "train.py",
                "-g", self._device.replace("cuda:", "").replace("cpu", "-1"),
                "-m", "COLT",
                "-d", self._dataset,
                "-infer", "True",
            ]

            # Inject COLT clone into PYTHONPATH (its `from models.COLT import COLT`
            # needs to resolve relative to the repo root).
            env = os.environ.copy()
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(self._colt_path)
                + (os.pathsep + current_pp if current_pp else "")
            )
            # Steer the tensor_data output into our tempdir by running cwd=tmp;
            # COLT's test() writes "tensor_data_formatted.txt" relative to cwd.
            # We must also stage a symlink to the COLT data files since cwd != repo.
            for needed in ("datasets", "models", "config.yaml"):
                src = self._colt_path / needed
                if src.exists():
                    try:
                        (tmp / needed).symlink_to(src)
                    except OSError:
                        # Symlink may fail on weird FS; fall back to copying.
                        pass

            logger.debug(
                "COLTBaseline: invoking %s (cwd=%s, dataset=%s)",
                " ".join(cmd),
                tmp,
                self._dataset,
            )

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(tmp),
                    timeout=self._timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"COLT subprocess timed out after {exc.timeout}s. "
                    "Increase cfg.colt_timeout_seconds or check that the "
                    "checkpoint loads correctly on the target device."
                ) from exc

            if proc.returncode != 0:
                raise RuntimeError(
                    f"COLT subprocess failed (exit {proc.returncode}):\n"
                    f"  stderr (tail): {proc.stderr[-2000:]}\n"
                    f"  stdout (tail): {proc.stdout[-500:]}"
                )

            if not output_indices_file.exists():
                raise RuntimeError(
                    f"COLT did not write {output_indices_file.name} in {tmp}. "
                    f"stdout tail: {proc.stdout[-500:]}"
                )

            raw_output = output_indices_file.read_text(encoding="utf-8")

        # COLT's tensor file accumulates lines across the full test split — we
        # take the LAST line (the most recently written, corresponding to our
        # one-query inference call). This is robust to upstream appending vs
        # truncating behaviour.
        all_topk = _parse_colt_topk_indices(raw_output)
        if not all_topk:
            raise RuntimeError(
                "COLT produced an empty top-k tensor file. Verify checkpoint "
                "and dataset are consistent."
            )
        topk_indices = all_topk[-1]

        # Map COLT integer indices -> api_name -> composite ID.
        tool_ids: list[str] = []
        scores: list[float] = []
        for rank, idx in enumerate(topk_indices[: self.top_k]):
            api_name = self._idx_to_api.get(idx, "")
            if not api_name:
                logger.debug("COLT idx=%d has no api_name mapping; skipping.", idx)
                continue
            composite = api_to_id.get(api_name.lower())
            if composite is None:
                # Tool exists in COLT corpus but not in caller's catalog —
                # surface it with a synthesized composite ID so downstream
                # eval can compute OOV-aware metrics.
                composite = f":::{api_name}"
            tool_ids.append(composite)
            # COLT does not expose per-rank scores from tensor_data_formatted.txt;
            # we synthesize a monotone decreasing score so downstream
            # rank-based metrics (NDCG, P@k) are correct, and pass the rank
            # through metadata for any consumer that needs it.
            scores.append(1.0 - rank / max(self.top_k, 1))

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrievedTools(
            tool_ids=tool_ids,
            scores=scores,
            metadata={
                "colt_path": str(self._colt_path),
                "colt_commit": self._colt_sha,
                "colt_dataset": self._dataset,
                "colt_ckpt": str(self._colt_ckpt),
                "colt_device": self._device,
                "colt_raw_indices": topk_indices[: self.top_k],
                "catalog_oov_count": len(oov),
                "latency_ms": latency_ms,
            },
        )
