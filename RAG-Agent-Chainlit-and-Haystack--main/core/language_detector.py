from typing import Tuple
from core.logging import get_logger

logger = get_logger(__name__)

class LanguageDetector:
    """
    Detects the primary language of the text and decides if MinerU should be used.
    """
    def __init__(self):
        self._logger = logger.bind(component="language_detector")

    def detect_and_route(self, text: str) -> Tuple[str, bool]:
        """
        Detects language and returns (language_code, use_mineru).
        MinerU is typically better for CJK (Chinese, Japanese, Korean).
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple containing the detected language code and a boolean indicating
            if the MinerU parser should be used instead of Docling.
        """
        if not text:
            return "en", False
            
        # Basic heuristic for CJK characters
        cjk_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u30ff' or '\uac00' <= char <= '\ud7af')
        
        # If more than 5% of characters are CJK, classify as CJK and suggest MinerU
        total_chars = len(text.strip())
        if total_chars > 0 and (cjk_char_count / total_chars) > 0.05:
            self._logger.info("detected_cjk_language", cjk_ratio=cjk_char_count/total_chars)
            # Defaulting to 'zh' as a generic CJK marker for logging purposes
            return "zh", True
            
        return "en", False
