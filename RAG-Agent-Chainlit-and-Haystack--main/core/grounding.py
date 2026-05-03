"""
Grounding Verification Engine

Implements dual-verification for answer grounding:
1. Pre-generation: Similarity threshold filtering
2. Post-generation: LLM-based fact verification

This is the core anti-hallucination mechanism. Every answer
passes through both stages before reaching the user.

Design decisions:
- Threshold-based hard filtering (no retrieval below 0.65)
- LLM-based soft verification (check claims against context)
- Configurable faithfulness threshold (default 0.75)
- Graceful degradation on verification failures
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from core.models import RetrievalResult, RefusalReason
from core.generator import LLMGenerator
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GroundingDecision:
    """Result of the grounding verification process."""

    is_grounded: bool
    faithfulness_score: float
    verified: bool
    reason: Optional[str] = None
    supported_claims: Optional[list[str]] = None
    unsupported_claims: Optional[list[str]] = None


class GroundingEngine:
    """
    Production grounding verification engine.

    Enforces strict document grounding through a two-stage
    verification pipeline.
    """

    def __init__(
        self,
        generator: LLMGenerator,
        similarity_threshold: float = 0.65,
        faithfulness_threshold: float = 0.75,
        min_context_chunks: int = 1,
    ) -> None:
        """
        Initialize the grounding engine.

        Args:
            generator: LLM generator for verification calls.
            similarity_threshold: Minimum retrieval similarity.
            faithfulness_score: Minimum faithfulness score to accept.
            min_context_chunks: Minimum chunks needed for an answer.
        """
        self._generator = generator
        self._similarity_threshold = similarity_threshold
        self._faithfulness_threshold = faithfulness_threshold
        self._min_context_chunks = min_context_chunks
        self._logger = logger.bind(component="grounding")

    def check_pre_generation(
        self,
        retrieval_results: list[RetrievalResult],
    ) -> GroundingDecision:
        """
        Pre-generation grounding check.

        Verifies that retrieval returned sufficient, relevant context
        before attempting generation.

        Args:
            retrieval_results: Chunks retrieved for the query.

        Returns:
            GroundingDecision indicating whether to proceed.
        """
        # No results at all
        if not retrieval_results:
            self._logger.info(
                "pre_check_failed",
                reason="no_retrieval_results",
            )
            return GroundingDecision(
                is_grounded=False,
                faithfulness_score=0.0,
                verified=True,
                reason="No relevant content found in the document.",
            )

        # Check if top result meets threshold
        top_score = retrieval_results[0].score
        if top_score < self._similarity_threshold:
            self._logger.info(
                "pre_check_failed",
                reason="below_threshold",
                top_score=top_score,
                threshold=self._similarity_threshold,
            )
            return GroundingDecision(
                is_grounded=False,
                faithfulness_score=top_score,
                verified=True,
                reason=(
                    f"Best match score ({top_score:.3f}) is below "
                    f"threshold ({self._similarity_threshold})."
                ),
            )

        # Sufficient context found
        self._logger.info(
            "pre_check_passed",
            num_results=len(retrieval_results),
            top_score=top_score,
        )
        return GroundingDecision(
            is_grounded=True,
            faithfulness_score=top_score,
            verified=True,
        )

    def check_post_generation(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
    ) -> GroundingDecision:
        """
        Post-generation grounding verification.

        Uses LLM-based fact-checking to verify every claim
        in the generated answer is supported by the context.

        Args:
            answer: Generated answer text.
            retrieval_results: Context chunks used for generation.

        Returns:
            GroundingDecision with verification details.
        """
        # Handle refusal responses — no verification needed
        if "I cannot answer" in answer:
            self._logger.info("post_check_skipped", reason="refusal_response")
            return GroundingDecision(
                is_grounded=True,
                faithfulness_score=1.0,
                verified=True,
                reason="Refusal response — no verification needed.",
            )

        self._logger.info("post_check_started")

        try:
            verification = self._generator.verify_grounding(
                answer=answer,
                context_chunks=retrieval_results,
            )

            faithfulness = verification.get("faithfulness_score", 0.0)
            verdict = verification.get("verdict", "FAIL")
            supported = verification.get("supported_claims", [])
            unsupported = verification.get("unsupported_claims", [])

            is_pass = (
                verdict == "PASS"
                and faithfulness >= self._faithfulness_threshold
            )

            self._logger.info(
                "post_check_completed",
                faithfulness_score=faithfulness,
                verdict=verdict,
                is_pass=is_pass,
                num_supported=len(supported),
                num_unsupported=len(unsupported),
            )

            return GroundingDecision(
                is_grounded=is_pass,
                faithfulness_score=faithfulness,
                verified=True,
                reason=(
                    None
                    if is_pass
                    else f"Faithfulness score ({faithfulness:.3f}) below "
                    f"threshold ({self._faithfulness_threshold}). "
                    f"{len(unsupported)} unsupported claims found."
                ),
                supported_claims=supported,
                unsupported_claims=unsupported,
            )

        except Exception as e:
            self._logger.error(
                "post_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # On verification failure, allow the answer through
            # but flag it as unverified
            return GroundingDecision(
                is_grounded=True,
                faithfulness_score=0.0,
                verified=False,
                reason="Verification failed — answer passed by default.",
            )

    def should_refuse(
        self,
        retrieval_results: list[RetrievalResult],
        grounding_decision: GroundingDecision,
    ) -> Optional[RefusalReason]:
        """
        Determine if the system should refuse to answer.

        Args:
            retrieval_results: Chunks retrieved for the query.
            grounding_decision: Result of grounding checks.

        Returns:
            RefusalReason if refusal is warranted, None otherwise.
        """
        if not retrieval_results:
            return RefusalReason.NO_RELEVANT_CONTENT

        if retrieval_results[0].score < self._similarity_threshold:
            return RefusalReason.BELOW_THRESHOLD

        if (
            grounding_decision.verified
            and not grounding_decision.is_grounded
        ):
            return RefusalReason.VERIFICATION_FAILED

        return None
