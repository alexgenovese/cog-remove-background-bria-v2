# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import torch, os, sys
from PIL import Image
from diffusers.utils import load_image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# sys.path.append("./script")
# from download_weights import start_download

MODEL_CACHE = "model-cache/"
DEVICE = "cuda"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # start_download()
        
        self.model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True, cache_dir=MODEL_CACHE)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.model.to(DEVICE)

        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def process(self, image):
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)

        return image
    
    def get_image(self, image: str):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]
    

    def predict(
        self,
        image: Path = Input(description="Remove background from this image"),
    ) -> Path:
        print("Start inference")
        image = self.get_image(image)
        transparent = self.process(image)
        
        save_path = "/tmp/output.png"
        transparent.save(save_path)

        return Path(save_path)
