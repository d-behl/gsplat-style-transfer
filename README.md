## Setting up virtual environment

Use CUDA-11.8.0 in ~/.bashrc (see WSI guide for bashrc setup)

Use `cgproj` virtual environment:
``` source ./cgproj/bin/activate ```

## Dataset

Currently using the `lerf` dataset for testing. Lerf scenes are stored in the lerf_ovs directory.

Datasets preprocessed for usage in nerfstudio are stored in the `data` directory. Currently only the figurines scenes is preprocessed and stored there. See preprocess_example.sh and nerfstudio docs to process other multi-view image scenes.

## Training a 3D Gaussian scene from preprocessed dataset

To train a gaussian splatting model for the figurines scene, run:
```ns-train splatfacto-big --data data/figurines_processed --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.use_scale_regularization True```

Use `--viewer.make-share-url True` to make a publicly shareable link to the viewer. (makes training slightly slower)

## Viewing a trained 3D Gaussian scene

To view the trained gaussian splatting model for the figurines scene, run:
```ns-viewer --load-config outputs/figurines_processed/splatfacto/2024-12-06_145554/config.yml```

Use `--viewer.make-share-url True` to make a publicly shareable link to the viewer. (makes training slightly slower)

To create a high-resolution render video of the scene, see the render menu of the viewer to export a nice video for the demo.

## Segmentation + Style Transfer

Trained gsplat models for scenes are stored in the `outputs` directory. Use the config.yml from the desired run in the first cell of the `gs_segmentation.ipynb` notebook for the segmentation & style transfer part.

The segmentation part of the notebook takes as input the Gaussian splatting model and input_points on the object from a given render/viewpoint and returns `final_mask` vairable that contains a mask over the Gaussians indicating which Gaussians belong to the selected object.

The style transfer section takes in the gsplat model along with the mask and a style image. It returns the Gaussians with style applied only to the masked Gaussians. (Note: new Gaussians not yet compatible with Nerfstudio viewer.) Still some bugs to be fixed here but mostly works.

## (Buggy) Viewing the styled Gaussians

Since styled Gaussians aren't yet usable in the Nerfstudio viewer, I currently tested them with another viser-based viewer.

Use:
```python simple_viewer.py --ckpt```

## To test on new scenes/objects/styles:

- Train Gaussian Spltting model for new scenes and use that config file in the jupyter notebook
OR
- Select `input_point` variable to point to another object in the scene
OR
- Select a different style image to use in the notebook


