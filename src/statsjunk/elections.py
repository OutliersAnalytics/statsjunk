from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field


class FragmentationResult(BaseModel):
    """Laakso-Taagepera effective number of options (candidates or parties)."""

    effective_number: float = Field(ge=1)
    largest_share: float = Field(ge=0, le=1)
    n_options: int = Field(ge=1)


def compute_laakso_taagepera(votes: Sequence[float]) -> FragmentationResult:
    """Laakso-Taagepera effective number of options for a vote distribution.

    ``N = 1 / Σ pᵢ²`` where ``pᵢ`` is option ``i``'s share of the total. It is
    ``1`` when a single option takes every vote and approaches the number of
    options as the vote splits evenly — a fragmentation / dispersion measure.

    Parameters
    ----------
    votes : Sequence[float]
        Vote count per option (candidate or party). Zeros are allowed;
        options with zero votes don't affect ``N``.

    Returns
    -------
    FragmentationResult
        ``effective_number`` (Laakso-Taagepera ``N``), ``largest_share`` (the
        leading option's vote share), and ``n_options`` (options with at
        least one vote).

    Raises
    ------
    ValueError
        If:
        - no options are provided
        - any vote count is negative
        - the votes sum to zero

    """
    votes = np.asarray(votes, dtype=float)

    if votes.size == 0:
        raise ValueError("At least one option is required.")
    if np.any(votes < 0):
        raise ValueError("Vote counts must be non-negative.")

    total = votes.sum()
    if total == 0:
        raise ValueError("Total votes must be positive.")

    shares = votes / total
    hhi = float(np.sum(shares**2))

    return FragmentationResult(
        effective_number=1.0 / hhi,
        largest_share=float(shares.max()),
        n_options=int(np.sum(votes > 0)),
    )
