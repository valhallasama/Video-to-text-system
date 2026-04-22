"""Spell correction and word completion for OCR output."""

from typing import Dict, List, Optional


class SpellCorrector:
    """Correct common OCR errors and complete partial words."""
    
    def __init__(self):
        # Medical/anatomical terms dictionary
        self.medical_terms = {
            'anatomy', 'anatomical', 'cardiac', 'medial', 'medially', 'lateral',
            'superior', 'inferior', 'anterior', 'posterior',
            'clockwise', 'counterclockwise', 'counter-clockwise',
            'rotate', 'rotation', 'tilt', 'slide', 'sweep', 'sweeps',
            'circular', 'hold', 'recording', 'appears', 'moving',
            'slowly', 'slow', 'quickly', 'fast', 'gently',
            'until', 'make', 'tail', 'more', 'toward', 'indicator',
            'sternum', 'closer'
        }
        
        # Common OCR errors - comprehensive list
        self.corrections = {
            # Rotate variations
            'rotat': 'rotate',
            'roae': 'rotate',
            'rotae': 'rotate',
            # Counter variations
            'counte': 'counter',
            'couner': 'counter',
            'couner-': 'counter-clockwise',
            'couner-clockwise': 'counter-clockwise',
            'counter-': 'counter-clockwise',
            # Clockwise
            'clockwi': 'clockwise',
            'cloghwise': 'clockwise',
            'cloghwis': 'clockwise',
            'clockhwise': 'clockwise',
            # Slowly/slow
            'slowl': 'slowly',
            'slov': 'slowly',
            'slow': 'slowly',  # Normalize to 'slowly'
            'sow': 'slow',
            # Circular
            'circula': 'circular',
            'greular': 'circular',
            'greula': 'circular',
            'grcular': 'circular',
            'circlar': 'circular',
            # Sweep/sweeps
            'sweey': 'sweeps',
            'swe': 'sweeps',
            'sweep': 'sweeps',  # Normalize to plural
            # Until
            'unil': 'until',
            'unl': 'until',
            'untl': 'until',
            'pnt': 'until',
            'unt': 'until',
            'tnt': 'until',
            # More
            'mor': 'more',
            # Moving
            'mong': 'moving',
            'movir': 'moving',
            'movin': 'moving',
            'mownp': 'moving',
            # Appears
            'appears': 'appears',
            'appeas': 'appears',
            'apears': 'appears',
            'ae': 'appears',
            'ee': 'appears',
            # Make
            'mae': 'make',
            'mak': 'make',
            # Tactile (should be removed - not in vocabulary)
            'tactile': '',
            # Other common errors
            'dyn': '',
            'lt': '',
            'll': '',
            't': '',
            'yt': '',
            'kwise': '',
            'c': '',
            'ect': '',
            'acta': '',
            'iii': '',
            'i': '',
            'wey': '',
            # Common garbage from poor OCR
            'halal': '',
            'rh': '',
            'amas': '',
            'maaan': '',
            'rl': '',
            'dt': '',
            'ln': '',
            'al': '',
            'fee': '',
            'mh': '',
            'be': '',
            'es': '',
            'at': 'at',  # Keep 'at' as valid preposition
            'ha': '',
            'ait': '',
            'net': '',
            'kt': '',
            're': '',
            'basa': '',
            'rahal': '',
            'pte': '',
            'nel': '',
            'in': '',
            'et': '',
            'oe': '',
            'ht': '',
            'mei': '',
            'bmn': '',
            'bnn': '',
            'eik': '',
            'eo': '',
            'aoa': '',
            'he': '',
            'hana': '',
            'mamas': '',
            'rte': '',
            'mel': '',
            'about': '',
            'a': '',  # Single letter artifacts
            'pee': '',
            'elk': '',
            'seen': '',
            'ot': '',
            # Tail
            'ail': 'tail',
            'tal': 'tail',
            # Anatomy
            'anaomy': 'anatomy',
            'anatomv': 'anatomy',
            'anatom': 'anatomy',
            'anaumy': 'anatomy',
            # Recording
            'recordir': 'recording',
            'recordin': 'recording',
            # Toward (missing 't')
            'oward': 'toward',
            'towar': 'toward',
            # Indicator (missing 't')
            'indicaor': 'indicator',
            'indicato': 'indicator',
            # Sternum variations
            'sernum': 'sternum',
            'stemum': 'sternum',
            'sterum': 'sternum',
            'stemun': 'sternum',
            # Medially
            'medial': 'medially',
            'mediall': 'medially',
            # Closer
            'cr': 'closer',
            'clo': 'closer',
            'close': 'closer',
            'closr': 'closer',
            # Patient position text
            'paien': 'patient',
            'patien': 'patient',
            'paent': 'patient',
            'piion': 'position',
            'posiion': 'position',
            'positon': 'position',
            'heir': 'their',
            'thei': 'their',
            'lef': 'left',
            'righ': 'right',
            'lying': 'lying',
            'lyng': 'lying',
            # Side
            'sid': 'side',
            # On
            'on': 'on',
            # Fragments to remove - only clearly invalid artifacts
            'ews': '',
            'proocol': '',
        }
    
    def correct(self, text: str) -> str:
        """Correct spelling and complete partial words."""
        if not text:
            return text
        
        # Direct correction
        if text.lower() in self.corrections:
            return self.corrections[text.lower()]
        
        # Partial word completion
        completed = self._complete_word(text.lower())
        if completed:
            return completed
        
        return text
    
    def _complete_word(self, partial: str) -> Optional[str]:
        """Complete partial word using dictionary."""
        if len(partial) < 3:
            return None
        
        # Find words that start with this partial
        candidates = [
            word for word in self.medical_terms
            if word.startswith(partial) and len(word) > len(partial)
        ]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Find closest match
        for word in self.medical_terms:
            if self._similarity(partial, word) > 0.8:
                return word
        
        return None
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        
        # Simple character overlap ratio
        common = sum(1 for c in s1 if c in s2)
        return common / max(len(s1), len(s2))
    
    def correct_phrase(self, words: List[str]) -> List[str]:
        """Correct a list of words."""
        corrected = []
        for word in words:
            corrected_word = self.correct(word)
            if corrected_word:  # Only add non-empty corrections
                corrected.append(corrected_word)
        return corrected
