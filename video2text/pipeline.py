"""Main video-to-text pipeline orchestration."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .core import (
    FrameExtractor,
    Preprocessor,
    ROIDetector,
    OCREngine,
    TextPostprocessor,
    TemporalAggregator,
    InstructionParser
)
from .utils import ConfigLoader, Visualizer


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_index: int
    timestamp_sec: float
    raw_text: Optional[str]
    cleaned_text: Optional[str]
    confidence: float
    ocr_results: List[Dict[str, Any]]
    structured_instruction: Optional[Dict[str, Any]]


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    video_path: str
    total_frames_processed: int
    global_instruction: Optional[str]
    structured_global_instruction: Optional[Dict[str, Any]]
    frame_results: List[FrameResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "total_frames_processed": self.total_frames_processed,
            "global_instruction": self.global_instruction,
            "structured_global_instruction": self.structured_global_instruction,
            "frame_results": [asdict(fr) for fr in self.frame_results]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class VideoToTextPipeline:
    """Main pipeline for extracting instructions from ultrasound guidance videos."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        
        self.frame_extractor = FrameExtractor(
            fps=config.get("frame_extraction", {}).get("fps", 5.0),
            adaptive=config.get("frame_extraction", {}).get("adaptive", False),
            skip_similar=config.get("frame_extraction", {}).get("skip_similar", True),
            similarity_threshold=config.get("frame_extraction", {}).get("similarity_threshold", 0.95)
        )
        
        self.preprocessor = Preprocessor(config.get("preprocessing", {}))
        self.roi_detector = ROIDetector(config.get("roi", {}))
        self.ocr_engine = OCREngine(config.get("ocr", {}))
        self.text_postprocessor = TextPostprocessor(config.get("postprocessing", {}))
        self.temporal_aggregator = TemporalAggregator(config.get("temporal", {}))
        self.instruction_parser = InstructionParser(config.get("instruction_parsing", {}))
        
        self.save_debug = config.get("output", {}).get("save_debug_images", False)
        self.debug_dir = Path(config.get("output", {}).get("debug_output_dir", "debug_output"))
        
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "VideoToTextPipeline":
        """Create pipeline from configuration file."""
        config = ConfigLoader.load(config_path)
        return cls(config)
    
    def process_video(self, video_path: str) -> PipelineResult:
        """Process entire video and extract instructions."""
        frames = self.frame_extractor.extract(video_path)
        
        frame_results: List[FrameResult] = []
        
        for frame in frames:
            frame_result = self._process_frame(frame)
            frame_results.append(frame_result)
            
            self.temporal_aggregator.add_frame_result(
                frame_result.cleaned_text,
                frame_result.confidence
            )
        
        global_instruction = self.temporal_aggregator.get_current_instruction()
        
        structured_global = None
        if global_instruction:
            best_conf = max((fr.confidence for fr in frame_results if fr.cleaned_text == global_instruction), default=0.0)
            structured_global = self.instruction_parser.parse(global_instruction, best_conf).to_dict()
        
        return PipelineResult(
            video_path=video_path,
            total_frames_processed=len(frame_results),
            global_instruction=global_instruction,
            structured_global_instruction=structured_global,
            frame_results=frame_results
        )
    
    def _process_frame(self, frame) -> FrameResult:
        """Process a single frame through the pipeline."""
        preprocessed = self.preprocessor.process(frame.image)
        
        rois = self.roi_detector.detect(preprocessed)
        
        all_ocr_results = []
        best_text = None
        best_confidence = 0.0
        
        for roi in rois:
            roi_image = roi.crop(preprocessed)
            
            if self.save_debug:
                Visualizer.save_debug_frame(
                    roi_image, 
                    self.debug_dir, 
                    frame.index, 
                    "roi_crop"
                )
            
            ocr_results = self.ocr_engine.recognize(roi_image)
            
            for ocr_result in ocr_results:
                all_ocr_results.append({
                    "text": ocr_result.text,
                    "confidence": ocr_result.confidence,
                    "bbox": ocr_result.bbox
                })
                
                if ocr_result.confidence > best_confidence:
                    best_confidence = ocr_result.confidence
                    best_text = ocr_result.text
        
        cleaned_text = None
        if best_text:
            cleaned_text = self.text_postprocessor.process(best_text)
        
        structured = None
        if cleaned_text:
            structured = self.instruction_parser.parse(cleaned_text, best_confidence).to_dict()
        
        return FrameResult(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            raw_text=best_text,
            cleaned_text=cleaned_text,
            confidence=best_confidence,
            ocr_results=all_ocr_results,
            structured_instruction=structured
        )
