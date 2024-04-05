from typing import Any
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from matplotlib import pyplot as plt
from iris.utils.visualisation import IRISVisualizer
from iris.io import dataclasses as iris_dc


class ResultVisualizer(Callback, Algorithm):

    class Parameters(Algorithm.Parameters):
        pass

    __parameters_type__ = Parameters

    def __init__(self) -> None:
        super().__init__()
        self.vis = IRISVisualizer()

    def run(self, result):
        # Visualize the result of different nodes
        # !todo - implement some more nodes
        if isinstance(result, iris_dc.SegmentationMap):
            try:
                plt.imshow(result.predictions, interpolation="nearest")
                self.vis.plot_segmentation_map(result)
            except Exception as e:
                print(e)

        try:
            if isinstance(result, tuple) and isinstance(result[0], iris_dc.GeometryMask):
                self.vis.plot_geometry_mask(geometry_mask=result[0])

        except Exception as e:
            print(e)

        plt.show()
        return result

    def on_execute_end(self, result: Any) -> None:
        return self.run(result)
