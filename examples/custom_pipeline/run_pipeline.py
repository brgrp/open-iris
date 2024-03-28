import iris
import wget
import json
import cv2
import os, sys

print(f"IRIS version: {iris.__version__}")

iris_image_path = "iris.png"
if not os.path.isfile(iris_image_path):
    image_url = "https://wld-ml-ai-data-public.s3.amazonaws.com/public-iris-images/example_orb_image_1.png"
    wget.download(image_url, iris_image_path)

# wip: implement callbacks for all nodes
nodes_to_show = ["segmentation", "segmentation_binarization"]

with open("custom_pipeline_conf.json") as json_data:
    custom_pipeline_conf = json.load(json_data)

    for node in custom_pipeline_conf.get("pipeline", None):
        # to handle empty callbacks - fixed
        # node["callbacks"] = None
        if node.get("name", "") in nodes_to_show:
            node["callbacks"] = [{"class_name": "callbacks.ResultVisualizer", "params": {}}]
            print(node)

    iris_pipeline = iris.IRISPipeline(config=dict(custom_pipeline_conf))

    img_pixels = cv2.imread("iris.png", cv2.IMREAD_GRAYSCALE)
    result = iris_pipeline(img_data=img_pixels, eye_side="left")

    if result["error"] is None:
        print(json.dumps(result["metadata"], indent=2))
    else:
        sys.exit(-1)
