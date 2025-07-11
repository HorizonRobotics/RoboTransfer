import imageio
import numpy as np


def save_images_to_mp4(images, output_path, fps=30):
    frames = [np.array(frame) for frame in images]
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
