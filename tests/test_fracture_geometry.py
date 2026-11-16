import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fracture_geometry import fit_surface_sections, prepare_surface, ridge_maps, stack_operator_maps, trace_fracture


def test_stacked_layers_reject_one_sided_brightness_edge():
    image = np.full((144, 144), 180, np.uint8)
    image[:, 72:] = 90
    fused, layers = stack_operator_maps(image, ridge_maps(image))
    assert not trace_fracture(fused).skeleton.any()
    assert set(layers) == {"morphology", "opposing_edges", "canny"}


def test_fracture_graph_recovers_t_and_y_and_fits_a_line_profile():
    tests = ([(26, 72), (118, 72), (72, 120)], [(72, 24), (117, 100), (27, 100)])
    for index, arms in enumerate(tests):
        image = np.full((144, 144), 180, np.uint8)
        for point in arms:
            cv2.line(image, (72, 72), point, 65, 2, cv2.LINE_AA)
        fused, _ = stack_operator_maps(image, ridge_maps(image))
        trace = trace_fracture(fused)
        assert len(trace.junctions) == 1
        assert len(trace.endpoints) == 3
        assert trace.geometry[6+index] < trace.geometry[7-index]
        widths = fit_surface_sections(image, fused["angle"], trace.skeleton)
        assert np.isfinite(widths).all()
        assert widths[2] > 0.5


def test_crop_before_normalization_retains_the_local_surface():
    original = np.zeros((540, 960, 3), np.uint8)
    original[150:330, 370:590] = (80, 170, 210)
    crop = prepare_surface(Image.fromarray(original), [0.4, 0.3, 0.6, 0.6], quality=60)
    assert min(crop.shape[:2]) == 192
    assert abs(float(crop[:, :, 1].mean()) - 170) < 3
