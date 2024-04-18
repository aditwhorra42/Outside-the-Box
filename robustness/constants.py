from typing import List, Optional
from pydantic import BaseModel

class PerturbationType:
    NORM_BOUNDED = "norm_bounded"
    FGSM = "FGSM"


class ExampleType:
    KNOWN = "known"
    UNKNOWN = "unknown"


class PerturbationInfo(BaseModel):
    perturbation_type: str
    eps: Optional[int]


class AdversarialExample(BaseModel):
    original_image_path: str
    pertubed_image_path: str
    label: int
    example_type: str
    model_confidnce: float
    monitor_confidences: List[float]
    image_distance: float

    class Config:
        arbitrary_types_allowed = True


class GeneratedExamples(BaseModel):
    examples: List[AdversarialExample]
    perturbation_info: PerturbationInfo
    min_dist: Optional[float]
    norm_min_dist: float
    adv_ratio: float
    norm_adv_ratio: float
    num_misclassifications_for_iteration: List[int]
    num_adversarial_for_iteration: List[int]
