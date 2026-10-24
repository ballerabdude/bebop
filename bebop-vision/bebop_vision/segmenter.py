"""Instance-segment data structure returned by the SAM 3.1 concept segmenter.

``sam3_concepts`` builds these; the teacher-side tools read ``label``,
``mask`` and ``confidence``.
"""


class InstanceSegment:
    __slots__ = ("bbox", "confidence", "class_id", "label", "mask")

    def __init__(self, bbox, confidence, class_id, label, mask=None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.label = label
        self.mask = mask
