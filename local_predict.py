# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from downloadweights import start_download

MODEL_CACHE = "model-cache/"
DEVICE = "mps"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start_download()

        self.model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True, cache_dir=MODEL_CACHE)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.model.to(DEVICE)
        self.model.eval()


    def predict(
        self,
        image: Path = Input(description="Remove background from this image"),
    ) -> Path:
        """Run a single prediction on the model"""
        image = Image.open(image)
        
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        input_images = transform_image(image).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        
        save_path = "/tmp/output.png"
        image.save(save_path)

        return Path(save_path)
