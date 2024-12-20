# Prediction interface for Cog
from cog import BasePredictor, Path
import torch, os, sys
from diffusers.utils import load_image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

sys.path.append("./script")
from download_weights import start_download

MODEL_CACHE = "model-cache/"
DEVICE = "mps"



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start_download()
        
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


    def predict(
        self,
        image="https://fffiloni-diffusers-image-outpaint.hf.space/file=/tmp/gradio/e6846698832579693cacb09bc4ac8c8e0d3e7ec4ef6dda430fee13b05146c512/example_3.jpg"
    ) -> Path:
        """Run a single prediction on the model"""
        print("Start inference")
        image = load_image(image)
        image = image.convert("RGB")
        transparent = self.process(image)
        
        save_path = "/tmp/output.png"
        transparent.save(save_path)

        return Path(save_path)


if __name__ == "__main__":
    pred = Predictor()
    pred.setup()
    pred.predict()