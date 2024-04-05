import sys
from io import BytesIO

import cv2
import numpy as np
import requests

import iris

# 0. Load pipeline (example from NB)

pipeline_config = {
    "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.0.0"},
    "pipeline": [
        {
            "name": "segmentation",
            "algorithm": {"class_name": "iris.MultilabelSegmentation", "params": {}},
            "inputs": [{"name": "image", "source_node": "input"}],
            "callbacks": [],
        },
        {
            "name": "segmentation_binarization",
            "algorithm": {"class_name": "iris.MultilabelSegmentationBinarization", "params": {}},
            "inputs": [{"name": "segmentation_map", "source_node": "segmentation"}],
            "callbacks": [],
        },
        {
            "name": "vectorization",
            "algorithm": {"class_name": "iris.ContouringAlgorithm", "params": {}},
            "inputs": [{"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 0}],
            "callbacks": [],
        },
        {
            "name": "specular_reflection_detection",
            "algorithm": {"class_name": "iris.SpecularReflectionDetection", "params": {}},
            "inputs": [{"name": "ir_image", "source_node": "input"}],
            "callbacks": [],
        },
        {
            "name": "interpolation",
            "algorithm": {"class_name": "iris.ContourInterpolation", "params": {}},
            "inputs": [{"name": "polygons", "source_node": "vectorization"}],
            "callbacks": [],
        },
        {
            "name": "distance_filter",
            "algorithm": {"class_name": "iris.ContourPointNoiseEyeballDistanceFilter", "params": {}},
            "inputs": [
                {"name": "polygons", "source_node": "interpolation"},
                {"name": "geometry_mask", "source_node": "segmentation_binarization", "index": 1},
            ],
            "callbacks": [],
        },
        {
            "name": "eye_orientation",
            "algorithm": {"class_name": "iris.MomentOfArea", "params": {}},
            "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
            "callbacks": [],
        },
        {
            "name": "eye_center_estimation",
            "algorithm": {"class_name": "iris.BisectorsMethod", "params": {}},
            "inputs": [{"name": "geometries", "source_node": "distance_filter"}],
            "callbacks": [],
        },
        {
            "name": "smoothing",
            "algorithm": {"class_name": "iris.Smoothing", "params": {}},
            "inputs": [
                {"name": "polygons", "source_node": "distance_filter"},
                {"name": "eye_centers", "source_node": "eye_center_estimation"},
            ],
            "callbacks": [],
        },
        {
            "name": "geometry_estimation",
            "algorithm": {
                "class_name": "iris.FusionExtrapolation",
                "params": {
                    "circle_extrapolation": {"class_name": "iris.LinearExtrapolation", "params": {"dphi": 0.703125}},
                    "ellipse_fit": {"class_name": "iris.LSQEllipseFitWithRefinement", "params": {"dphi": 0.703125}},
                    "algorithm_switch_std_threshold": 3.5,
                },
            },
            "inputs": [
                {"name": "input_polygons", "source_node": "smoothing"},
                {"name": "eye_center", "source_node": "eye_center_estimation"},
            ],
            "callbacks": [],
        },
        {
            "name": "pupil_to_iris_property_estimation",
            "algorithm": {"class_name": "iris.PupilIrisPropertyCalculator", "params": {}},
            "inputs": [
                {"name": "geometries", "source_node": "geometry_estimation"},
                {"name": "eye_centers", "source_node": "eye_center_estimation"},
            ],
            "callbacks": [],
        },
        {
            "name": "offgaze_estimation",
            "algorithm": {"class_name": "iris.EccentricityOffgazeEstimation", "params": {}},
            "inputs": [{"name": "geometries", "source_node": "geometry_estimation"}],
            "callbacks": [],
        },
        {
            "name": "occlusion90_calculator",
            "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 90.0}},
            "inputs": [
                {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                {"name": "eye_orientation", "source_node": "eye_orientation"},
                {"name": "eye_centers", "source_node": "eye_center_estimation"},
            ],
            "callbacks": [],
        },
        {
            "name": "occlusion30_calculator",
            "algorithm": {"class_name": "iris.OcclusionCalculator", "params": {"quantile_angle": 30.0}},
            "inputs": [
                {"name": "noise_mask", "source_node": "segmentation_binarization", "index": 1},
                {"name": "extrapolated_polygons", "source_node": "geometry_estimation"},
                {"name": "eye_orientation", "source_node": "eye_orientation"},
                {"name": "eye_centers", "source_node": "eye_center_estimation"},
            ],
            "callbacks": [],
        },
        {
            "name": "noise_masks_aggregation",
            "algorithm": {"class_name": "iris.NoiseMaskUnion", "params": {}},
            "inputs": [
                {
                    "name": "elements",
                    "source_node": [
                        {"name": "segmentation_binarization", "index": 1},
                        {"name": "specular_reflection_detection"},
                    ],
                }
            ],
            "callbacks": [],
        },
        {
            "name": "normalization",
            "algorithm": {"class_name": "iris.PerspectiveNormalization", "params": {}},
            "inputs": [
                {"name": "image", "source_node": "input"},
                {"name": "noise_mask", "source_node": "noise_masks_aggregation"},
                {"name": "extrapolated_contours", "source_node": "geometry_estimation"},
                {"name": "eye_orientation", "source_node": "eye_orientation"},
            ],
            "callbacks": [],
        },
        {
            "name": "filter_bank",
            "algorithm": {
                "class_name": "iris.ConvFilterBank",
                "params": {
                    "filters": [
                        {
                            "class_name": "iris.GaborFilter",
                            "params": {
                                "kernel_size": [41, 21],
                                "sigma_phi": 7,
                                "sigma_rho": 6.13,
                                "theta_degrees": 90.0,
                                "lambda_phi": 28.0,
                                "dc_correction": True,
                                "to_fixpoints": True,
                            },
                        },
                        {
                            "class_name": "iris.GaborFilter",
                            "params": {
                                "kernel_size": [17, 21],
                                "sigma_phi": 2,
                                "sigma_rho": 5.86,
                                "theta_degrees": 90.0,
                                "lambda_phi": 8,
                                "dc_correction": True,
                                "to_fixpoints": True,
                            },
                        },
                    ],
                    "probe_schemas": [
                        {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                        {"class_name": "iris.RegularProbeSchema", "params": {"n_rows": 16, "n_cols": 256}},
                    ],
                },
            },
            "inputs": [{"name": "normalization_output", "source_node": "normalization"}],
            "callbacks": [],
        },
        {
            "name": "encoder",
            "algorithm": {"class_name": "iris.IrisEncoder", "params": {}},
            "inputs": [{"name": "response", "source_node": "filter_bank"}],
            "callbacks": [],
        },
        {
            "name": "bounding_box_estimation",
            "algorithm": {"class_name": "iris.IrisBBoxCalculator", "params": {}},
            "inputs": [
                {"name": "ir_image", "source_node": "input"},
                {"name": "geometry_polygons", "source_node": "geometry_estimation"},
            ],
            "callbacks": [],
        },
    ],
}


# 0. Add callbacks to pipeline - set callback for selected nodes
nodes_to_run_callback = ["segmentation", "segmentation_binarization"]

for node in pipeline_config.get("pipeline", None):
    if node.get("name", "") in nodes_to_run_callback:
        node["callbacks"] = [{"class_name": "result_visualizer.ResultVisualizer", "params": {}}]
        print(f"Callback ResultVisualizer added to {node['name']}.")

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline(config=dict(pipeline_config))

# 2. Load IR image of an eye
url = "https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png"
try:
    response = requests.get(url)
    assert response.status_code == 200, "Error: URL is not reachable."

    # Convert the bytes data to a NumPy array
    nparr = np.frombuffer(BytesIO(response.content).getvalue(), np.uint8)
    # Decode the image using OpenCV's imread function
    image_data = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

except Exception as e:
    sys.exit(e)

# 3. Perform inference with custom callback: ResultVisualizer -> shows the result of the selected nodes
output = iris_pipeline(img_data=image_data, eye_side="left")
if output["error"] is None:
    print(output["metadata"])
else:
    sys.exit(f"Error: {output}")
