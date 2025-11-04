import os
import argparse
from datasets import Dataset
from PIL import Image
import numpy as np
import cv2
import io
from tqdm import tqdm
import glob

def decompress_image(image_bytes):
    """Decompresses a PNG image from bytes."""
    if image_bytes is None:
        return None
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img



def create_training_frame_dict(sample, views, depth_type, background_dir, target_size, debug_data):
    """Creates a single video frame by combining different views."""

    out_dict = {}
    out_dict['depth_images'] = []
    out_dict['normal_images'] = []    
    out_dict['camera_images'] = []    
    out_dict['reference_images'] = []          
    out_dict['foreground_images'] = []

    for view in views:
        # --- Load Images from dataset sample ---
        rgb_bytes = sample.get("{}_camera_image".format(view))
        normal_bytes = sample.get("{}_normal_image".format(view))
        depth_bytes = sample.get("{}_{}_image".format(view, depth_type))
        # --- Handle Background Image Loading ---
        background_bytes = None
        if background_dir:
            scene_id = sample.get('scene_id')
            if scene_id:
                # Assuming the background is the first frame's background for the entire scene
                bg_path = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "0000_background", "0000.png")
                if os.path.exists(bg_path):
                    with open(bg_path, 'rb') as f:
                        background_bytes = f.read()
                else:
                    # Fallback or log warning if specific background not found
                    pass
                tmppath = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg")
                fg_paths = glob.glob(os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg"))
                

        # Fallback to background image from dataset if not loaded from external dir
        if background_bytes is None:
            background_bytes = sample.get("{}_background_image".format(view))
        
        foreground_imgs = []
        for fg_path in fg_paths:
            if os.path.exists(fg_path):
                with open(fg_path, 'rb') as f:
                    foreground_bytes = f.read()
                foreground_img = Image.open(io.BytesIO(foreground_bytes)).convert("RGB")
                foreground_imgs.append(foreground_img)
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs = foreground_imgs[:3]
        
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB") if rgb_bytes else None
        normal_img = decompress_image(normal_bytes)
        depth_img = decompress_image(depth_bytes)
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB") if background_bytes else None
        # --- Handle Missing Images and Resize ---
        w, h = target_size
        # print("w, h: ", w, h)
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)          
        depth_placeholder = np.zeros((h, w), dtype=np.uint8)    
                
        if rgb_img is not None:
            # h, w = rgb_img.height, rgb_img.width
            rgb_img = np.array(rgb_img).astype(np.uint8)
   
            if rgb_img is not None:
                rgb_img = cv2.resize(rgb_img, (w, h)).astype(np.uint8)
            else:
                rgb_img = placeholder
                
            if normal_img is not None:
                # print("normal_img: ", normal_img.dtype, normal_img.min(), normal_img.max())
                normal_img = cv2.resize(normal_img, (w, h)).astype(np.uint8)
            else:
                normal_img = placeholder
            if depth_img is not None:
                min_valid_depth = 0.1
                max_valid_depth = 4.0
                # 如果是uint16格式，转换为米为单位的float
                depth_array = cv2.resize(depth_img, (w, h))
                
                # print("depth_array: ", depth_array.dtype, depth_array.min(), depth_array.max())
                
                if depth_array.dtype == np.uint16:
                    depth_array = depth_array.astype(float) / 1000.0  # 转换为米
                depth_array[depth_array < min_valid_depth] = min_valid_depth
                depth_array[depth_array > max_valid_depth] = max_valid_depth
                depth_img = depth_array
                
                # print("in depth_img.shape: ", depth_img.shape)
                # Normalize depth image from uint16 to uint8
                # depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # depth_img = cv2.cvtColor(cv2.resize(depth_img_normalized, (w, h)), cv2.COLOR_GRAY2BGR)
            else:
                depth_img = depth_placeholder
                # print("out depth_img.shape: ", depth_img.shape)
                
            
            if background_img is not None:
                background_img = cv2.resize(np.array(background_img), (w, h)).astype(np.uint8)
            else:
                background_img = placeholder

            # --- Add Titles to Images ---
            def add_text(img, text):
                # Ensure image is writeable
                img = np.ascontiguousarray(img)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return img
            # rgb_img = add_text(rgb_img, "RGB")
            # normal_img = add_text(normal_img, "Normal")
            # depth_img = add_text(depth_img, "Depth")
            # background_img = add_text(background_img, "Background")
            # foreground_img0 = add_text(foreground_img0, "foreground_img0")
            # foreground_img1 = add_text(foreground_img1, "foreground_img1")
            # foreground_img2 = add_text(foreground_img2, "foreground_img2")
            # --- Combine Images for the View ---
            # view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img, foreground_img0, foreground_img1]) #, foreground_img2])
            # view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img])
            
            # view_frames.append(view_frame)
        else:
            depth_img = depth_placeholder
            normal_img = placeholder
            rgb_img = placeholder
            background_img = placeholder
            print("rgb_img none: ", debug_data)
            
            
        out_dict['depth_images'].append(depth_img)
        out_dict['normal_images'].append(normal_img)   
        out_dict['camera_images'].append(rgb_img)    
        out_dict['reference_images'].append(background_img)
        out_dict['foreground_images'].append(foreground_imgs)            
    return out_dict



def create_visualization_frame_dict(sample, views, depth_type, background_dir, target_size):
    """Creates a single video frame by combining different views."""

    out_dict = {}
    out_dict['depth_images'] = []
    out_dict['normal_images'] = []    
    out_dict['camera_images'] = []    
    out_dict['reference_images'] = []          
    out_dict['foreground_images'] = []

    for view in views:
        # --- Load Images from dataset sample ---
        rgb_bytes = sample.get("{}_camera_image".format(view))
        normal_bytes = sample.get("{}_normal_image".format(view))
        depth_bytes = sample.get("{}_{}_image".format(view, depth_type))
        # --- Handle Background Image Loading ---
        background_bytes = None
        if background_dir:
            scene_id = sample.get('scene_id')
            if scene_id:
                # Assuming the background is the first frame's background for the entire scene
                bg_path = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "0000_background", "0000.png")
                if os.path.exists(bg_path):
                    with open(bg_path, 'rb') as f:
                        background_bytes = f.read()
                else:
                    # Fallback or log warning if specific background not found
                    pass
                tmppath = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg")
                fg_paths = glob.glob(os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg"))
                

        # Fallback to background image from dataset if not loaded from external dir
        if background_bytes is None:
            background_bytes = sample.get("{}_background_image".format(view))
        
        foreground_imgs = []
        for fg_path in fg_paths:
            if os.path.exists(fg_path):
                with open(fg_path, 'rb') as f:
                    foreground_bytes = f.read()
                foreground_img = Image.open(io.BytesIO(foreground_bytes)).convert("RGB")
                foreground_imgs.append(foreground_img)
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs.append(np.zeros((224,224,3)))
        # foreground_imgs = foreground_imgs[:3]
        
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB") if rgb_bytes else None
        normal_img = decompress_image(normal_bytes)
        depth_img = decompress_image(depth_bytes)
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB") if background_bytes else None
        # --- Handle Missing Images and Resize ---
        if rgb_img is not None:
            # h, w = rgb_img.height, rgb_img.width
            rgb_img = np.array(rgb_img).astype(np.uint8)
            w, h = target_size
            # print("w, h: ", w, h)
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)            
            if rgb_img is not None:
                rgb_img = cv2.resize(rgb_img, (w, h)).astype(np.uint8)
            else:
                rgb_img = placeholder
                
            if normal_img is not None:
                normal_img = cv2.resize(normal_img, (w, h)).astype(np.uint8)
            else:
                normal_img = placeholder
            if depth_img is not None:
                # Normalize depth image from uint16 to uint8
                depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_img = cv2.cvtColor(cv2.resize(depth_img_normalized, (w, h)), cv2.COLOR_GRAY2BGR)
            else:
                depth_img = placeholder
            if background_img is not None:
                background_img = cv2.resize(np.array(background_img), (w, h)).astype(np.uint8)
            else:
                background_img = placeholder

            # --- Add Titles to Images ---
            def add_text(img, text):
                # Ensure image is writeable
                img = np.ascontiguousarray(img)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return img
            # rgb_img = add_text(rgb_img, "RGB")
            # normal_img = add_text(normal_img, "Normal")
            # depth_img = add_text(depth_img, "Depth")
            # background_img = add_text(background_img, "Background")
            # foreground_img0 = add_text(foreground_img0, "foreground_img0")
            # foreground_img1 = add_text(foreground_img1, "foreground_img1")
            # foreground_img2 = add_text(foreground_img2, "foreground_img2")
            # --- Combine Images for the View ---
            # view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img, foreground_img0, foreground_img1]) #, foreground_img2])
            # view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img])
            
            # view_frames.append(view_frame)
            
            out_dict['depth_images'].append(depth_img)
            out_dict['normal_images'].append(normal_img)   
            out_dict['camera_images'].append(rgb_img)    
            out_dict['reference_images'].append(background_img)
            out_dict['foreground_images'].append(foreground_imgs)
            
    # if not view_frames:
        # return None
    # --- Combine All Views into a Single Frame ---
    # final_frame = np.vstack(view_frames)
    # return final_frame
    return out_dict



def create_visualization_frame(sample, views, depth_type, background_dir=None):
    """Creates a single video frame by combining different views."""
    view_frames = []
    for view in views:
        # --- Load Images from dataset sample ---
        rgb_bytes = sample.get("{}_camera_image".format(view))
        normal_bytes = sample.get("{}_normal_image".format(view))
        depth_bytes = sample.get("{}_{}_image".format(view, depth_type))
        # --- Handle Background Image Loading ---
        background_bytes = None
        if background_dir:
            scene_id = sample.get('scene_id')
            if scene_id:
                # Assuming the background is the first frame's background for the entire scene
                bg_path = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "0000_background", "0000.png")
                if os.path.exists(bg_path):
                    with open(bg_path, 'rb') as f:
                        background_bytes = f.read()
                else:
                    # Fallback or log warning if specific background not found
                    pass
                tmppath = os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg")
                fg_paths = glob.glob(os.path.join(background_dir, str(scene_id), "{}_mask".format(view), "*_crop_*.jpg"))
                

        # Fallback to background image from dataset if not loaded from external dir
        if background_bytes is None:
            background_bytes = sample.get("{}_background_image".format(view))
        
        foreground_imgs = []
        for fg_path in fg_paths:
            if os.path.exists(fg_path):
                with open(fg_path, 'rb') as f:
                    foreground_bytes = f.read()
                foreground_img = Image.open(io.BytesIO(foreground_bytes)).convert("RGB")
                foreground_imgs.append(foreground_img)
        foreground_imgs.append(np.zeros((224,224,3)))
        foreground_imgs.append(np.zeros((224,224,3)))
        foreground_imgs.append(np.zeros((224,224,3)))
        foreground_imgs = foreground_imgs[:3]
        
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB") if rgb_bytes else None
        normal_img = decompress_image(normal_bytes)
        depth_img = decompress_image(depth_bytes)
        background_img = Image.open(io.BytesIO(background_bytes)).convert("RGB") if background_bytes else None
        # --- Handle Missing Images and Resize ---
        if rgb_img is not None:
            h, w = rgb_img.height, rgb_img.width
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_img = np.array(rgb_img).astype(np.uint8)
            if normal_img is not None:
                normal_img = cv2.resize(normal_img, (w, h)).astype(np.uint8)
            else:
                normal_img = placeholder
            if depth_img is not None:
                # Normalize depth image from uint16 to uint8
                depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_img = cv2.cvtColor(cv2.resize(depth_img_normalized, (w, h)), cv2.COLOR_GRAY2BGR)
            else:
                depth_img = placeholder
            if background_img is not None:
                background_img = cv2.resize(np.array(background_img), (w, h)).astype(np.uint8)
            else:
                background_img = placeholder
            foreground_img0 = cv2.resize(np.array(foreground_imgs[0]), (w, h)).astype(np.uint8)
            foreground_img1 = cv2.resize(np.array(foreground_imgs[1]), (w, h)).astype(np.uint8)
            foreground_img2 = cv2.resize(np.array(foreground_imgs[2]), (w, h)).astype(np.uint8)
            
            # --- Add Titles to Images ---
            def add_text(img, text):
                # Ensure image is writeable
                img = np.ascontiguousarray(img)
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return img
            rgb_img = add_text(rgb_img, "RGB")
            normal_img = add_text(normal_img, "Normal")
            depth_img = add_text(depth_img, "Depth")
            background_img = add_text(background_img, "Background")
            foreground_img0 = add_text(foreground_img0, "foreground_img0")
            foreground_img1 = add_text(foreground_img1, "foreground_img1")
            foreground_img2 = add_text(foreground_img2, "foreground_img2")
            # --- Combine Images for the View ---
            view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img, foreground_img0, foreground_img1]) #, foreground_img2])
            # view_frame = np.hstack([rgb_img, normal_img, depth_img, background_img])
            
            view_frames.append(view_frame)
    if not view_frames:
        return None
    # --- Combine All Views into a Single Frame ---
    final_frame = np.vstack(view_frames)
    return final_frame


def main():
    parser = argparse.ArgumentParser(description="Create a video from a Hugging Face Parquet dataset.")
    parser.add_argument("--dataset_path", type=str, default='/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_0_2000/dataset_0_49.parquet', help="Path to the dataset directory.")
    parser.add_argument("--output_path", type=str, default='./data/agibot/visualization_376_654009.mp4', help="Path to save the output video.")
    parser.add_argument("--background_dir", type=str, default='/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_0_2000/fg_bg_0_2000', help="Path to the external directory for background images.")
    parser.add_argument("--scene_id", type=str, default='376_654009', help="The scene_id to visualize.")
    parser.add_argument("--start_frame_id", type=int, default=0, help="Start frame_id of samples.")
    parser.add_argument("--end_frame_id", type=int, default=None, help="End frame_id of samples (inclusive).")
    parser.add_argument("--views", type=str, default="hand_left,hand_right,head", help="Comma-separated views.")
    parser.add_argument("--depth_type", type=str, default='depth', help="Type of depth data.")
    parser.add_argument("--fps", type=int, default=10, help="FPS for the output video.")
    args = parser.parse_args()
    if not os.path.exists(args.dataset_path):
        print("Error: Dataset path not found at '{}'".format(args.dataset_path))
        return
    print("Loading dataset from: {}".format(args.dataset_path))
    full_dataset = Dataset.load_from_disk(args.dataset_path)
    print("Full dataset loaded. Total samples: {}".format(len(full_dataset)))
    # --- Diagnostic Step: Print unique scene_ids ---
    if 'scene_id' in full_dataset.column_names:
        unique_scene_ids = set(full_dataset['scene_id'])
        print("unique_scene_ids: ", unique_scene_ids)
        
        print("\n--- Found {} Unique Scene IDs ---".format(len(unique_scene_ids)))
        # Print a few examples
        for i, scene_id in enumerate(list(unique_scene_ids)[:10]):
            print("  - {} (Type: {})".format(scene_id, type(scene_id)))
        if len(unique_scene_ids) > 10:
            print("  ...")
        print("-------------------------------------\n")
    else:
        print("Warning: 'scene_id' column not found in the dataset.")
    # Filter by scene_id and sort by frame_id
    print("Filtering for scene_id: {}...".format(args.scene_id))
    scene_dataset = full_dataset.filter(lambda example: str(example['scene_id']) == str(args.scene_id), keep_in_memory=True)
    if len(scene_dataset) == 0:
        print("Error: No samples found for scene_id '{}'".format(args.scene_id))
        return
    # Sort in memory using pandas to avoid caching issues
    print("Sorting scene samples by frame_id...")
    scene_df = scene_dataset.to_pandas()
    scene_df = scene_df.sort_values('frame_id').reset_index(drop=True)
    scene_dataset = Dataset.from_pandas(scene_df)
    # Determine the slice of the dataset to process
    frame_ids = list(scene_dataset['frame_id'])
    start_index = 0
    if args.start_frame_id is not None:
        try:
            start_index = frame_ids.index(args.start_frame_id)
        except ValueError:
            print("Warning: start_frame_id {} not found in scene. Starting from the beginning.".format(args.start_frame_id))
    end_index = len(scene_dataset)
    if args.end_frame_id is not None:
        try:
            # Find the index and include it in the slice
            end_index = frame_ids.index(args.end_frame_id) + 1
        except ValueError:
            print("Warning: end_frame_id {} not found in scene. Processing until the end.".format(args.end_frame_id))
    if start_index >= end_index:
        print("Error: start_frame_id must be less than end_frame_id.")
        return
    dataset_slice = scene_dataset.select(range(start_index, end_index))
    print("Processing {} samples from frame_id {} to {}.".format(len(dataset_slice), frame_ids[start_index], frame_ids[end_index-1]))
    views_list = [v.strip() for v in args.views.split(',')]
    # --- Initialize Video Writer ---
    first_frame = create_visualization_frame(dataset_slice[0], views_list, args.depth_type, args.background_dir)
    if first_frame is None:
        print("Could not generate the first frame. Exiting.")
        return
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    video_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (w, h))
    # --- Generate and Write Frames ---
    for sample in tqdm(dataset_slice, desc="Creating video"):
        frame = create_visualization_frame(sample, views_list, args.depth_type, args.background_dir)
        if frame is not None:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    print("Video saved to: {}".format(args.output_path))
if __name__ == "__main__":
    main()