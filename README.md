# 🎨 BRIA 2.0 Background Remover

> Professional-grade background removal powered by BRIA AI's RMBG-2.0 model, implemented for Replicate using Cog

## 🚀 Features

- State-of-the-art background removal
- Handles multiple image categories
- Professional-grade segmentation
- Optimized for commercial content

## 🛠️ Quick Start

### Setup
```bash
# Download pre-trained weights
cog run script/download-weights

# Run a prediction
cog predict -i image=@your-image.jpg
```

## 📊 Model Capabilities

| Content Type | Support Level |
|--------------|---------------|
| Objects | 45.11% |
| People & Objects | 25.24% |
| People Only | 17.35% |
| Mixed Content | 8.52% |
| Text | 2.52% |
| Animals | 1.89% |

## 🎯 Use Cases
- E-commerce product photography
- Professional content creation
- Digital advertising
- Gaming assets
- Stock image processing

## 📸 Example Results

**Before**

Input Image

**After**

Output Image

## ⚙️ Technical Specifications
- Architecture: BiRefNet
- Input Resolution: Up to 1024x1024
- Output Format: PNG with transparency
- Processing: CUDA-optimized

## 🤝 Credits
Powered by [BRIA AI's RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)[1]

---
*Built with Cog for seamless deployment on Replicate*

Citations:
[1] https://huggingface.co/briaai/RMBG-2.0
