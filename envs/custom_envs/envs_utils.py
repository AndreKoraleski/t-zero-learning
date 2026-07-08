"""Helpers for Gymnasium vector env step infos (autoreset / RecordEpisodeStatistics)."""

from __future__ import annotations

from typing import Any

import numpy as np


def episode_completions_from_vector_infos(
    terminations: np.ndarray | Any,
    truncations: np.ndarray | Any,
    infos: dict[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    """
    Episodic stats for envs that finished on this vector ``step``.

    Gymnasium ``AutoresetMode.SAME_STEP`` (Meta-World MT10+ training) places terminal
    ``RecordEpisodeStatistics`` data under ``infos["final_info"]``, not at the root.
    Single-sub-env vectors often expose root ``_episode`` / ``episode`` instead.

    Returns:
        (episodic_returns, episodic_lengths, successes). ``successes`` is empty when the
        env does not expose a ``success`` key in the terminal info; otherwise it has
        one float per completed episode, aligned with the return/length lists.
    """
    done = np.logical_or(
        np.asarray(terminations, dtype=bool),
        np.asarray(truncations, dtype=bool),
    )
    returns: list[float] = []
    lengths: list[float] = []
    successes: list[float] = []

    fi = infos.get("final_info")
    if isinstance(fi, dict) and "episode" in fi:
        ep = fi["episode"]
        if isinstance(ep, dict) and "r" in ep:
            final_mask = np.asarray(infos.get("_final_info", done), dtype=bool)
            mask = np.logical_and(done, final_mask)
            if mask.any():
                er = np.asarray(ep["r"], dtype=np.float64).reshape(-1)
                el = (
                    np.asarray(ep["l"], dtype=np.float64).reshape(-1)
                    if "l" in ep
                    else np.zeros_like(er)
                )
                m = np.asarray(mask, dtype=bool).reshape(-1)
                k = min(m.size, er.size)
                m = m[:k]
                returns = er[:k][m].astype(np.float64, copy=False).tolist()
                lengths = el[:k][m].astype(np.float64, copy=False).tolist()
                if "success" in fi:
                    sr = np.asarray(fi["success"], dtype=np.float64).reshape(-1)[:k]
                    successes = sr[m].astype(np.float64, copy=False).tolist()
                return returns, lengths, successes

    if "_episode" in infos and "episode" in infos:
        episode_mask = np.asarray(infos["_episode"], dtype=bool).reshape(-1)
        if not episode_mask.any():
            return [], [], []
        ep_r = np.asarray(infos["episode"]["r"], dtype=np.float64).reshape(-1)
        ep_l = np.asarray(infos["episode"]["l"], dtype=np.float64).reshape(-1)
        k = min(episode_mask.size, ep_r.size)
        m = episode_mask[:k]
        returns = ep_r[:k][m].astype(np.float64, copy=False).tolist()
        lengths = ep_l[:k][m].astype(np.float64, copy=False).tolist()
        if "success" in infos:
            su = np.asarray(infos["success"], dtype=np.float64).reshape(-1)[:k]
            successes = su[m].astype(np.float64, copy=False).tolist()

    return returns, lengths, successes
