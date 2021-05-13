# VIZIR Project


## Abstract

This project aims to leverage the UCDIG paper from Taigman et. al, 2017 to develop a set of tools that geenrate equivalent IR domain images from Visual range images. This set of tools allows for development of large IR training datasets from existing Visual datasets.

## Rationale and Justification

IR image generation, i.e. using an IR camera is extremely expensive undertaking as compared to visual range images. ALmost every phone in use on the world now contains HD resolution (leading to 8k UHD in flogship models) camera. Datasets get built through social media tagging of images performed either by photographers or using existing ML infrastructure. There is no equivalent technological and social solution available for IR image generation. Therefore a tool that can automate conversion of large visual datasets into IR datasets is needed to bridge this gap.

## Why do we need IR images anyway? 

IR is historically an EM radiation range of interest for astronomy, meteorology, and defense. It is now expanding into consumer domain rapidly through LIDARs, IR enhanced night photography, home security and automation etc. We forsee a future where integration of both visual range and IR range cameras will happen for most cameras available on market as better IR photodetector materials with sufficient performance using passive cooling becomes a well developed technology. We expect that unlike astronomy or meteorology, these cameras will be brodband (i.e. not focused on a narrow range of frequency considered as a signature of particular phsyical phenomena being imaged). Therefore machine vision infrasturtcure needs to be able to perform high quality sensor fusion and inferencing capabilities using the IR range images to nearly as high degree of performance as visual range images.

## What is under the hood?

We adapt the UCDIG paper from Taigman et. al, (ICLR 2017) to form the basis of our codebase. This paper develops a composite generator function that considers both source (visual) and target (IR) domain images together to develop loss functions. This Generator also uses a two step approach using a feature extractor (f) and a generator (g) function that separates the two functions in two different trainable networks. The feature extractor network is used again to ensure constancy of the target domain images when they go through the combined f o g o f combination and is considered as one of the loss functions. The details can be found in the paper itself available in preprint form at https://arxiv.org/pdf/1611.02200.pdf. The codebase is developed in TensorFlow 2.0 using Keras. The models use the functional form for network definitions. We test the images using the CAMEL datatset developed at GeorgiaTech. The approach is kept flexible enough to be able to handle other datasets in future.

## Caveats and sundries

This is a codebase in development, so *do not* use it in its current form. We will metion when the code goes in the alpha stage at which point it will be ready for use by anyone else than me. The code uses a BSD license.
