"""Parse instruction text into structured commands."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class StructuredInstruction:
    """Structured probe movement instruction."""
    action: Optional[str] = None
    direction: Optional[str] = None
    speed: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InstructionParser:
    """Parse natural language instructions into structured commands."""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        
        self.action_keywords = config.get("actions", {})
        self.direction_keywords = config.get("directions", {})
        self.speed_keywords = config.get("speeds", {})
    
    def parse(self, text: str, confidence: float = 1.0) -> StructuredInstruction:
        """Parse text into structured instruction."""
        if not self.enabled or not text:
            return StructuredInstruction(raw_text=text, confidence=confidence)
        
        text_lower = text.lower()
        
        action = self._extract_action(text_lower)
        direction = self._extract_direction(text_lower)
        speed = self._extract_speed(text_lower)
        
        return StructuredInstruction(
            action=action,
            direction=direction,
            speed=speed,
            raw_text=text,
            confidence=confidence
        )
    
    def _extract_action(self, text: str) -> Optional[str]:
        """Extract action from text."""
        for action, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return action
        return None
    
    def _extract_direction(self, text: str) -> Optional[str]:
        """Extract direction from text."""
        for direction, keywords in self.direction_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return direction
        return None
    
    def _extract_speed(self, text: str) -> Optional[str]:
        """Extract speed modifier from text."""
        for speed, keywords in self.speed_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return speed
        return None
