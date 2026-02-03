from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

def filter_detections(
    result,
    strategy: Literal['best', 'all'] = 'best',
    top_n: int = 1
):
    """
    Filter YOLO detection results based on strategy.
    
    Args:
        result: YOLO result object with boxes attribute
        strategy: 'best' returns highest confidence, 'all' returns all detections
        top_n: Number of top detections to return (only used with 'best' strategy)
    
    Returns:
        Filtered result object
        
    Note:
        This filtering is applied AFTER YOLO's built-in NMS.
        For single-object scenarios, 'best' strategy is recommended.
    """
    if strategy == 'all':
        return result
    
    if strategy == 'best' and len(result.boxes) > 0:
        # Get indices of top N detections by confidence
        if top_n == 1:
            max_idx = result.boxes.conf.argmax()
            result.boxes = result.boxes[max_idx.item()]
        else:
            # Get top N indices
            top_indices = result.boxes.conf.argsort(descending=True)[:top_n]
            result.boxes = result.boxes[top_indices]
    
    return result

def get_detection_info(result) -> Optional[dict]:
    """
    Extract detection information from YOLO result.
    
    Args:
        result: YOLO result object
        
    Returns:
        Dictionary with detection info or None if no detections
    """
    if len(result.boxes) == 0:
        return None
    
    box = result.boxes[0]
    return {
        "class": int(box.cls),
        "confidence": float(box.conf),
        "bbox": box.xywh.tolist()[0]  # [x_center, y_center, width, height]
    }
