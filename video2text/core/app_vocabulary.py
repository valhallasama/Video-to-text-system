"""Limited vocabulary constraint for ultrasound APP guidance system."""

from typing import List, Set, Optional
from difflib import get_close_matches


class APPVocabulary:
    """Constrain OCR output to known APP guidance vocabulary."""
    
    def __init__(self):
        # Complete vocabulary for ultrasound probe guidance
        self.vocabulary: Set[str] = {
            # Actions
            'make', 'rotate', 'turn', 'tilt', 'angle', 'slide', 'move',
            'press', 'push', 'hold', 'keep', 'maintain',
            'sweep', 'sweeps', 'fan', 'rock',
            
            # Directions
            'clockwise', 'counter-clockwise', 'counterclockwise',
            'left', 'right', 'up', 'down', 'upward', 'downward',
            'medial', 'lateral', 'superior', 'inferior',
            'anterior', 'posterior', 'cephalad', 'caudal',
            'tail', 'head',
            
            # Speeds/Modifiers
            'slow', 'slowly', 'quick', 'quickly', 'fast',
            'gently', 'firmly', 'slight', 'slightly',
            'more', 'less', 'very',
            
            # Descriptors
            'circular', 'linear', 'straight', 'curved',
            'small', 'large', 'wide', 'narrow',
            
            # Anatomical
            'anatomy', 'anatomical', 'structure', 'structures',
            'cardiac', 'heart', 'chamber', 'chambers',
            'valve', 'valves', 'vessel', 'vessels',
            
            # Instructions
            'until', 'while', 'when', 'if', 'then',
            'appears', 'visible', 'seen', 'clear',
            'moving', 'still', 'static',
            
            # Recording
            'hold', 'recording', 'record', 'capture',
            'save', 'clip', 'image',
            
            # Prepositions/Conjunctions
            'for', 'to', 'in', 'on', 'at', 'with',
            'and', 'or', 'the', 'a', 'an',
        }
        
        # Common OCR errors to vocabulary mapping
        self.error_corrections = {
            'rotat': 'rotate',
            'slowl': 'slowly',
            'circula': 'circular',
            'anatomv': 'anatomy',
            'counte': 'counter',
            'clockwi': 'clockwise',
            'sweey': 'sweep',
            'recordir': 'recording',
            'medail': 'medial',
            'laterai': 'lateral',
        }
    
    def constrain(self, word: str) -> Optional[str]:
        """Constrain word to vocabulary, return None if not valid."""
        if not word:
            return None
        
        word_lower = word.lower()
        
        # Direct match
        if word_lower in self.vocabulary:
            return word_lower
        
        # Known error correction
        if word_lower in self.error_corrections:
            return self.error_corrections[word_lower]
        
        # Fuzzy match (for partial words)
        matches = get_close_matches(word_lower, self.vocabulary, n=1, cutoff=0.85)
        if matches:
            return matches[0]
        
        # Check if it's a partial word that can be completed
        candidates = [v for v in self.vocabulary if v.startswith(word_lower) and len(v) > len(word_lower)]
        if len(candidates) == 1:
            return candidates[0]
        
        return None
    
    def filter_words(self, words: List[str]) -> List[str]:
        """Filter list of words to only valid vocabulary."""
        filtered = []
        for word in words:
            constrained = self.constrain(word)
            if constrained:
                filtered.append(constrained)
        return filtered
    
    def get_vocabulary_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.vocabulary)
    
    def add_word(self, word: str):
        """Add a new word to vocabulary (for customization)."""
        self.vocabulary.add(word.lower())
