# Documentation
Codebase for manuscript "Propagating spatio-temporal activity patterns across macaque motor cortex carry kinematic information"

To replicate the environment, you can either build your own image by using the Dockerfile_copy in the github repo (probably quicker, but their latest gpu environment might have changed thus not 100% the same as what I'm running now), or use the image I pushed to docker repo and build from there (probably slower to download but would be exactly as the environment I'm running): https://hub.docker.com/r/weiliang08/tf-custom-ipython

Main code: denoise_and_extractSpatialPatterns_demo.py  
Core autoencoder-related functions are in contractive_autoencoder_core.py  
Auxiliary functions are in contractive_autoencoder_supportive.py

Core autoencoder-related functions were adapted from this implementation- https://github.com/zaouk/contractive_autoencoders  
The original contractive autoencoder is proposed in this paper- https://icml.cc/2011/papers/455_icmlpaper.pdf
