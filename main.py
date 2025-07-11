import torch
from PIL import Image
from datasets import load_dataset
from robotransfer import RoboTransferPipeline
from robotransfer.utils.image_loading import load_images_from_dataset
from robotransfer.utils.save_video import save_images_to_mp4

def main():
    dataset = load_dataset("wkqin/robotransfer-example-dataset")

    pipe = RoboTransferPipeline.from_pretrained(
        "wkqin/robotransfer-high-resolution",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to("cuda")

    frames = []
    for i in range(0, len(dataset["train"]) - 30, 30):
        depth_guider_images, normal_guider_images, save_video = load_images_from_dataset(dataset["train"], frames_start=i, frames_end=i + 30)
        frames += pipe(
            image=Image.open("assets/example_ref_image/gray_grid_desk.png"),
            depth_guider_images=depth_guider_images,
            normal_guider_images=normal_guider_images,
            min_guidance_scale=1.,
            max_guidance_scale=3,
            height=384,
            width=640*3,
            num_frames=30,
            num_inference_steps=25,
        ).frames[0]
        if save_video:
            save_images_to_mp4(frames, "output_frames_final.mp4", fps=10)
        save_images_to_mp4(frames, "output_frames.mp4", fps=10)


if __name__ == "__main__":
    main()
