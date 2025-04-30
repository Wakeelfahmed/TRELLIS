import os
os.environ['SPCONV_ALGO'] = 'native'

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

base_dir = os.path.join(os.getcwd(), "Multi images dataset")
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subdir in subdirs:
    img_dir = os.path.join(base_dir, subdir)
    img_files = sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    if not img_files:
        continue

    print(f"Processing {subdir} with files:")
    for img in img_files:
        print(f"  {img}")

    images = [Image.open(img) for img in img_files]

    outputs = pipeline.run_multi_image(
        images,
        seed=1,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3
        },
    )

    output_dir = os.path.join(base_dir, f"output_{subdir}")
    os.makedirs(output_dir, exist_ok=True)

    outputs['mesh'][0].save(os.path.join(output_dir, "mesh.obj"))

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(os.path.join(output_dir, "model.glb"))

    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]

    imageio.mimsave(os.path.join(output_dir, "output.mp4"), video, fps=30)

    print(f"Processed {subdir}: Mesh, GLB and video saved in {output_dir}")
