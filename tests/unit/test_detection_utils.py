import pytest
import torch
from unittest.mock import MagicMock
from app.utils.detection import filter_detections, get_detection_info

@pytest.fixture
def mock_yolo_result():
    result = MagicMock()
    
    # helper to create a mock box at specific indices
    def create_boxes_mock(indices):
        # Convert indices to list for indexing if it's a tensor
        idx_list = indices.tolist() if hasattr(indices, 'tolist') else indices
        
        conf = torch.tensor([0.9, 0.7, 0.8])[idx_list]
        cls = torch.tensor([1, 1, 1])[idx_list]
        xywh = torch.tensor([[10, 10, 20, 20], [30, 30, 10, 10], [50, 50, 5, 5]])[idx_list]
        
        m = MagicMock()
        m.conf = conf
        m.cls = cls
        m.xywh = xywh
        # For indexing, if idx is a tensor, convert to list
        m.__getitem__ = MagicMock(side_effect=lambda idx: create_boxes_mock(
            (idx.tolist() if hasattr(idx, 'tolist') else [idx]) if isinstance(idx, (int, torch.Tensor)) else idx
        ))
        m.__len__ = MagicMock(return_value=len(idx_list))
        return m

    result.boxes = create_boxes_mock([0, 1, 2])
    return result

def test_filter_detections_best(mock_yolo_result):
    filtered = filter_detections(mock_yolo_result, strategy='best', top_n=1)
    # Best should be index 0 (0.9)
    assert len(filtered.boxes) == 1
    assert filtered.boxes.conf.item() == pytest.approx(0.9)

def test_filter_detections_top_n(mock_yolo_result):
    filtered = filter_detections(mock_yolo_result, strategy='best', top_n=2)
    # Top 2 should be indices 0 (0.9) and 2 (0.8)
    assert len(filtered.boxes) == 2
    assert filtered.boxes.conf[0].item() == pytest.approx(0.9)
    assert filtered.boxes.conf[1].item() == pytest.approx(0.8)

def test_filter_detections_all(mock_yolo_result):
    filtered = filter_detections(mock_yolo_result, strategy='all')
    assert len(filtered.boxes) == 3

def test_get_detection_info(mock_yolo_result):
    # Filter to 1 first
    filtered = filter_detections(mock_yolo_result, strategy='best', top_n=1)
    info = get_detection_info(filtered)
    assert info['class'] == 1
    assert info['confidence'] == pytest.approx(0.9)
    assert info['bbox'] == [pytest.approx(v) for v in [10.0, 10.0, 20.0, 20.0]]

def test_get_detection_info_empty():
    result = MagicMock()
    result.boxes = []
    assert get_detection_info(result) is None
