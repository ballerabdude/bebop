"""Defaults for the SAM 3.1 concept segmenter (bebop_vision/sam3_concepts.py).

Its callers (tools/bev_student_labels.py) pass the concepts and the device
explicitly; these are the fallbacks.
"""

DEVICE = "auto"
RECORD_CONCEPTS = ("floor", "wall", "person", "chair", "table", "door")
