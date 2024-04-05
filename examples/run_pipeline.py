import sys
from io import BytesIO

import cv2
import numpy as np
import requests

import iris

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()

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

# 3. Perform inference
# Options for the `eye_side` argument are: ["left", "right"]
output = iris_pipeline(img_data=image_data, eye_side="left")
if output["error"] is None:
    print(output["metadata"])
else:
    sys.exit(f"Error: {output}")
