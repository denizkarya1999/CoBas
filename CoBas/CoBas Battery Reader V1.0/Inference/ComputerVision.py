import cv2
import torch


class ComputerVision:
    """
    Computer Vision / PyTorch inference placeholder for CoBas_V1.

    This class currently does not perform image processing or inference.
    It only loads OpenCV and PyTorch and confirms they are available.

    PyTorch is forced to CPU mode.
    """

    def __init__(self):
        """
        Initialize the computer vision module and log library status.
        """

        self.device = torch.device("cpu")
        self.model = None

        self.log_library_status()

    def log_library_status(self):
        """
        Log OpenCV and PyTorch loading status.
        """

        print("========================================")
        print("CoBas_V1 Computer Vision Module")
        print("========================================")

        print("[INFO] OpenCV library loaded successfully.")
        print(f"[INFO] OpenCV version: {cv2.__version__}")

        print("[INFO] PyTorch library loaded successfully.")
        print(f"[INFO] PyTorch version: {torch.__version__}")

        print("[INFO] PyTorch device forced to CPU.")
        print(f"[INFO] Active device: {self.device}")

        print("========================================")

    def load_model(self, model_path=None):
        """
        Placeholder for future model loading.

        Currently does not load any model.
        """

        print("[INFO] load_model() called.")
        print("[INFO] No model loading is implemented yet.")

        return None

    def run_inference(self, frame):
        """
        Placeholder for future inference.

        Currently returns the original frame without changes.
        """

        return frame