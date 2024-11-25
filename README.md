# BRIA 2.0

This is an implementation of the [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@giraffa.webp

## Example

Input shoe image

![alt text](giraffa.jpg)

Output shoe image with background removed

![alt text](output.png)


## BRIA Background Removal v2.0 Model Card
RMBG v2.0 is our new state-of-the-art background removal model significantly improves RMBG v1.4. The model is designed to effectively separate foreground from background in a range of categories and image types.

