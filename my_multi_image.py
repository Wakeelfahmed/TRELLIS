import os
import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Load the pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Get all subdirectories in the current directory
base_dir = os.path.join(os.getcwd(), "Multi images dataset")
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subdir in subdirs:
    img_dir = os.path.join(base_dir, subdir)
    img_files = sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    if not img_files:
        continue

    print(f"Processing {subdir} with files:")  # Debugging output
    for img in img_files:
        print(f"  {img}")

    # Load images
    images = [Image.open(img) for img in img_files]

    # # Run the pipeline
    outputs = pipeline.run_multi_image(
        images,
        seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )

    # Create output directory
    output_dir = os.path.join(base_dir, f"output_{subdir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save mesh
    mesh_file = os.path.join(output_dir, "mesh.obj")
    outputs['mesh'][0].save(mesh_file)

    # Generate video
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]

    # Save video
    video_path = os.path.join(output_dir, "output.mp4")
    imageio.mimsave(video_path, video, fps=30)

    print(f"Processed {subdir}: Mesh and video saved in {output_dir}")
