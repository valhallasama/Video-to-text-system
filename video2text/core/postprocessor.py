"""Text postprocessing and cleaning."""

import re
from typing import Dict, Any, List


class TextPostprocessor:
    """Clean and normalize OCR output text."""
    
    def __init__(self, config: Dict[str, Any]):
        self.remove_special = config.get("remove_special_chars", True)
        self.normalize_ws = config.get("normalize_whitespace", True)
        self.lowercase = config.get("lowercase", True)
        self.spell_check = config.get("spell_check", False)
        self.blacklist = set(config.get("blacklist", []))
        
        # Initialize spell corrector if enabled
        self.spell_corrector = None
        if self.spell_check or config.get("spell_correction", {}).get("enabled", False):
            from .spell_corrector import SpellCorrector
            self.spell_corrector = SpellCorrector()
        
        # Initialize APP vocabulary constraint if enabled
        self.use_app_vocabulary = config.get("app_vocabulary", {}).get("enabled", False)
        self.app_vocab = None
        if self.use_app_vocabulary:
            from .app_vocabulary import APPVocabulary
            self.app_vocab = APPVocabulary()
    
    def process(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        result = text
        
        if self.remove_special:
            result = re.sub(r'[^\w\s\-]', ' ', result)
        
        if self.normalize_ws:
            result = re.sub(r'\s+', ' ', result).strip()
        
        if self.lowercase:
            result = result.lower()
        
        # Apply blacklist filtering first
        for blacklisted in self.blacklist:
            result = result.replace(blacklisted, '')
        
        result = result.strip()
        
        # Apply APP vocabulary constraint OR spell correction (not both)
        if self.app_vocab:
            # Try vocabulary constraint first
            constrained = self.app_vocab.constrain(result)
            if constrained:
                result = constrained
            else:
                # If not in vocab, try spell correction
                if self.spell_corrector:
                    corrected = self.spell_corrector.correct(result)
                    if corrected:
                        result = corrected
                    else:
                        result = ''  # Filter out if no correction found
        elif self.spell_corrector:
            # Just spell correction if vocab not enabled
            result = self.spell_corrector.correct(result)
        
        return result
