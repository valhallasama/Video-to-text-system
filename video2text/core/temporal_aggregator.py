"""Temporal aggregation for video-level stability."""

from typing import Dict, Any, List, Optional
from collections import defaultdict


class TemporalAggregator:
    """Aggregate OCR results across multiple frames."""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.method = config.get("method", "majority_vote")
        self.window_size = config.get("window_size", 5)
        self.min_confidence = config.get("min_confidence", 0.6)
        
        self.hysteresis_enabled = config.get("hysteresis", {}).get("enabled", True)
        self.hysteresis_threshold = config.get("hysteresis", {}).get("threshold", 0.8)
        
        self._current_instruction = None
        self._buffer: List[Dict[str, Any]] = []
    
    def add_frame_result(self, text: Optional[str], confidence: float):
        """Add a frame's OCR result to the buffer."""
        if text and confidence >= self.min_confidence:
            self._buffer.append({"text": text, "confidence": confidence})
        
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
    
    def get_current_instruction(self) -> Optional[str]:
        """Get the current aggregated instruction."""
        if not self.enabled or not self._buffer:
            return None
        
        if self.method == "majority_vote":
            return self._majority_vote()
        elif self.method == "confidence_weighted":
            return self._confidence_weighted()
        elif self.method == "moving_average":
            return self._moving_average()
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _majority_vote(self) -> Optional[str]:
        """Select most frequent text in buffer."""
        counts = defaultdict(float)
        
        for item in self._buffer:
            text = item["text"]
            conf = item["confidence"]
            counts[text] += conf
        
        if not counts:
            return None
        
        best_text = max(counts.items(), key=lambda x: x[1])[0]
        
        if self.hysteresis_enabled and self._current_instruction:
            if self._current_instruction in counts:
                current_score = counts[self._current_instruction]
                best_score = counts[best_text]
                
                if current_score >= best_score * self.hysteresis_threshold:
                    return self._current_instruction
        
        self._current_instruction = best_text
        return best_text
    
    def _confidence_weighted(self) -> Optional[str]:
        """Weight by confidence scores."""
        return self._majority_vote()
    
    def _moving_average(self) -> Optional[str]:
        """Simple moving average (same as majority for text)."""
        return self._majority_vote()
    
    def reset(self):
        """Clear buffer and state."""
        self._buffer.clear()
        self._current_instruction = None
