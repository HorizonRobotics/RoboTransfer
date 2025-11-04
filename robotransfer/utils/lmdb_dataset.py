import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile, PngImagePlugin
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from .. import utils
from .base_dataset import BaseDataset
from transformers import CLIPImageProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from .load_sample import create_visualization_frame_dict, create_training_frame_dict
import glob


class HuggingfaceNemoDataset(BaseDataset):
    def __init__(self, dataset_name_list, **kwargs):
        super(HuggingfaceNemoDataset , self).__init__(kwargs.get('config_path', None), dataset_name_list, kwargs.get('transform', None))
        # print("dataset_name: ", dataset_name)
        self.lendataset = 0
        dataset_list = []
        bg_list = []
        self.lendataset_list = []        
        
        bg_name_list = kwargs.get('bg_name')
        # dataset_name_list = [
        #       "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_20000_24000_100/*.parquet"
        # ]
        # bg_name_list = [
        #       '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_20000_24000'
        # ]
        
        print("dataset_name_list: ", dataset_name_list)
        print("bg_name_list: ", bg_name_list)
        
        debug_data_list = []
        
        for dataset_name_, bg_name in zip(dataset_name_list, bg_name_list):
            dataset_names = glob.glob(dataset_name_)

            for dataset_name in dataset_names:
                try:
                    dataset = load_from_disk(dataset_name)
                except:
                    print("failed loading: ", dataset_name)
                    continue
                unique_scene_ids = set(dataset['scene_id'])
                scene_dataset = dataset
                dataset_list.append(scene_dataset)
                bg_list.append(bg_name)
                
                self.lendataset += len(scene_dataset)
                self.lendataset_list.append(len(scene_dataset))
                debug_data_list.append(dataset_name)
            #     break
            # break
            
        self.debug_data_list = debug_data_list
        self.dataset_list = dataset_list
        self.bg_list = bg_list
        
        data_shuffle = kwargs.get('data_shuffle', True)
        self.frame_num = kwargs.get('frame_num', 30)
        
        self.multiview = kwargs.get('multiview', True)
        
        self.idx_list = [idx for idx in range(0, self.lendataset, self.frame_num)]
        
        print("multiview: ", self.multiview)
        print("frame_num: ", self.frame_num)

        model_path = "/horizon-bucket/robot_lab/users/jeff.wang/models/public/huggingface/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1"
        self.clip_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder='feature_extractor')
        
        self.target_size = kwargs.get('target_size', (192, 128))
    
        print("self.target_size: ", self.target_size)
        
        if data_shuffle:
            np.random.shuffle(self.idx_list)
            
    def __len__(self):
        return len(self.idx_list)

    def _get_data(self, index):
        data_dict = {}

        frames_start = self.idx_list[index]  ### 总的index
        
        ### 对应回每个分的index
        sum_lendataset = 0
        for dataset_idx, lendataset in enumerate(self.lendataset_list):
            sum_lendataset2 = sum_lendataset + lendataset
            if frames_start >= sum_lendataset and frames_start < sum_lendataset2:
                break
            sum_lendataset = sum_lendataset2
        
        dataset = self.dataset_list[dataset_idx]
        background_dir = self.bg_list[dataset_idx]
        debug_data = self.debug_data_list[dataset_idx]
        
        lendataset = self.lendataset_list[dataset_idx]
        frames_start = frames_start - sum_lendataset
        
        scene_id = dataset[frames_start]["scene_id"]

        temporal_depth_images = []
        temporal_normal_images = []
        temporal_camera_images = []
        
        colored_depth = False
        views ="hand_left,head,hand_right"
        views_list = [v.strip() for v in views.split(',')]

        # import cv2
        # output_path = './data/agibot/visualization_376_654009.mp4'
        # fps = 12
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        depth_type = 'depth'
        
        # background_dir = '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_0_2000/fg_bg_0_2000'
        
        # video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        # --- Generate and Write Frames ---
        for i in range(frames_start, frames_start+self.frame_num):
            if i >= lendataset:
                i = lendataset - 1
            # padding if scene_id has changed
            if dataset[i]["scene_id"] != scene_id:
                temporal_depth_images.append(temporal_depth_images[-1])
                temporal_normal_images.append(temporal_normal_images[-1])
                temporal_camera_images.append(temporal_camera_images[-1])
                continue
            
            sample = dataset[i]
            out_dict = create_training_frame_dict(sample, views_list, depth_type, background_dir, self.target_size, debug_data)

            
            depth_left_image, depth_middle_image, depth_right_image = out_dict['depth_images']
            normal_left_image, normal_middle_image, normal_right_image = out_dict['normal_images']
            camera_left_image, camera_middle_image, camera_right_image = out_dict['camera_images']
            
            if self.multiview:
                if colored_depth:
                    depth_image = np.concatenate(
                        [
                            np.array(depth_left_image) / 255,
                            np.array(depth_middle_image) / 255,
                            np.array(depth_right_image) / 255,
                        ],
                        axis=1,
                    )
                else:
                    depth_image = np.concatenate(
                        [
                            np.array(depth_left_image) ,
                            np.array(depth_middle_image) ,
                            np.array(depth_right_image) ,
                        ],
                        axis=1,
                    )
                    
                normal_image = np.concatenate(
                    [
                        np.array(normal_left_image) / 255,
                        np.array(normal_middle_image) / 255,
                        np.array(normal_right_image) / 255,
                    ],
                    axis=1,
                )
                camera_image = np.concatenate(
                    [
                        np.array(camera_left_image) / 255,
                        np.array(camera_middle_image) / 255,
                        np.array(camera_right_image) / 255,
                    ],
                    axis=1,
                )
            else:
                if colored_depth:
                    depth_image = np.array(depth_middle_image) / 255
                else:
                    depth_image = np.array(depth_middle_image)
                normal_image = np.array(normal_middle_image) / 255
                camera_image =  np.array(camera_middle_image) / 255
                
                
            temporal_depth_images.append(depth_image)
            temporal_normal_images.append(normal_image)
            temporal_camera_images.append(camera_image)
            
            if i == frames_start:
                reference_left_image, reference_middle_image, reference_right_image = out_dict['reference_images']
                object_reference_left_images, object_reference_middle_images, object_reference_right_images = out_dict['foreground_images']
                num_objs = len(object_reference_middle_images)

        MAX_NUM_OBJS = 10
        def resize_obj_list(images):
            new_images = []
            cnt = 0
            for _image in images:
                _image = _image.resize(size=self.target_size)
                _image = _image.convert('RGB')
                _image = np.array(_image)
                new_images.append(_image)
                cnt += 1
                if cnt == MAX_NUM_OBJS:
                    break
            while cnt < MAX_NUM_OBJS:
                _image0 = np.zeros((self.target_size[1], self.target_size[0], 3)).astype("uint8")
                new_images.append(_image0)
                cnt += 1
            return new_images
        
        object_reference_left_images = resize_obj_list(object_reference_left_images)
        object_reference_middle_images = resize_obj_list(object_reference_middle_images)
        object_reference_right_images = resize_obj_list(object_reference_right_images)
        
        temporal_ref_images = []

        if self.multiview:
            reference_image = np.concatenate(
                [
                    np.array(reference_left_image) / 255,
                    np.array(reference_middle_image) / 255,
                    np.array(reference_right_image) / 255,
                ],
                axis=1,
            )
        else:
            reference_image = np.array(reference_middle_image) / 255       
        
        temporal_ref_images.append(reference_image)
        temporal_depth_image = np.stack(temporal_depth_images, axis=0)
        
        if colored_depth:
            temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).permute(
                0, 3, 1, 2
            )   
        else:         
            temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(dim=1)
            # temporal_depth_image_pt = temporal_depth_image_pt.clip(
            #     min=0.1, max=2.0
            # )
            
        temporal_normal_image = np.stack(temporal_normal_images, axis=0)
        temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
            0, 3, 1, 2
        )
        temporal_normal_image_pt = 2 * temporal_normal_image_pt - 1

        temporal_camera_image = np.stack(temporal_camera_images, axis=0)
        temporal_camera_image_pt = torch.from_numpy(temporal_camera_image).permute(
            0, 3, 1, 2
        )
        temporal_camera_image_pt = 2 * temporal_camera_image_pt - 1

        temporal_ref_image = np.stack(temporal_ref_images, axis=0)
        temporal_ref_image_pt = torch.from_numpy(temporal_ref_image).permute(
            0, 3, 1, 2
        )
        temporal_ref_image_pt = 2 * temporal_ref_image_pt - 1

        temporal_obj_ref_middle_image = np.stack(object_reference_middle_images, axis=0) / 255
        temporal_obj_ref_middle_image_pt = torch.from_numpy(temporal_obj_ref_middle_image).permute(
            0, 3, 1, 2
        )
        temporal_obj_ref_middle_image_pt = 2 * temporal_obj_ref_middle_image_pt - 1
        
        data_dict = {}
        data_dict['concat'] = temporal_camera_image_pt
        data_dict['concat_depth'] = temporal_depth_image_pt
        data_dict['concat_normal'] = temporal_normal_image_pt
        data_dict['concat_ref_image'] = temporal_ref_image_pt
        data_dict['concat_ref_obj_image'] = temporal_obj_ref_middle_image_pt
        
        input_clip_image = _resize_with_antialiasing(data_dict['concat_ref_image'], (224, 224))
        input_clip_image = (input_clip_image + 1.0) / 2.0
        input_clip_image = self.clip_processor(
            images=input_clip_image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors='pt',
        ).pixel_values
        data_dict['concat_clip_image'] = input_clip_image

        input_clip_image = _resize_with_antialiasing(data_dict['concat_ref_obj_image'], (224, 224))
        input_clip_image = (input_clip_image + 1.0) / 2.0
        input_clip_image = self.clip_processor(
            images=input_clip_image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors='pt',
        ).pixel_values
        data_dict['concat_obj_clip_image'] = input_clip_image
        
        num_pad_objs = MAX_NUM_OBJS - num_objs
        encoder_attention_mask = torch.zeros((1, MAX_NUM_OBJS + 1), device=input_clip_image.device, dtype=input_clip_image.dtype)
        if num_pad_objs > 0:
            encoder_attention_mask[:, -num_pad_objs:] = float('-inf')
        data_dict['encoder_attention_mask'] = encoder_attention_mask
    
    
    
    
        temporal_camera_image_pt = (temporal_camera_image_pt + 1.0)/2 * 255
        temporal_normal_image_pt = (temporal_normal_image_pt + 1.0)/2 * 255
        temporal_depth_image_pt = (temporal_depth_image_pt + 1.0)/2   * 255
        temporal_ref_image_pt = (temporal_ref_image_pt + 1.0)/2   * 255
        temporal_obj_ref_middle_image_pt =  (temporal_obj_ref_middle_image_pt + 1.0)/2   * 255
        
        temporal_camera_image_pt = temporal_camera_image_pt.permute(
            0, 2, 3, 1
        )
        temporal_normal_image_pt = temporal_normal_image_pt.permute(
            0, 2, 3, 1
        )
        temporal_depth_image_pt = temporal_depth_image_pt.permute(
            0, 2, 3, 1
        )           
        temporal_ref_image_pt = temporal_ref_image_pt.permute(
            0, 2, 3, 1
        )   

        temporal_obj_ref_middle_image_pt = temporal_obj_ref_middle_image_pt.permute(
            0, 2, 3, 1
        )
        
        debug_vid_dir = "./debug_vid_dir_0919"
        os.makedirs(debug_vid_dir, exist_ok=True)
                
        # output_path = os.path.join(debug_vid_dir, f'./debug_{frames_start}_temporal_camera_image_pt.mp4')
        # save_images_to_mp4(temporal_camera_image_pt.numpy(), output_path, fps=2)
        
        # output_path = os.path.join(debug_vid_dir, f'./debug_{frames_start}_temporal_normal_image_pt.mp4')
        # save_images_to_mp4(temporal_normal_image_pt.numpy(), output_path, fps=2)

        # output_path = os.path.join(debug_vid_dir, f'./debug_{frames_start}_temporal_depth_image_pt.mp4')
        # save_images_to_mp4(temporal_depth_image_pt.numpy(), output_path, fps=2)
       
        # output_path = os.path.join(debug_vid_dir, f'./debug_{frames_start}_temporal_ref_image_pt.mp4')
        # save_images_to_mp4(temporal_ref_image_pt.numpy(), output_path, fps=2)

        # output_path = os.path.join(debug_vid_dir, f'./debug_{frames_start}_temporal_obj_ref_middle_image_pt.mp4')
        # save_images_to_mp4(temporal_obj_ref_middle_image_pt.numpy(), output_path, fps=2)
        
        # exit()
        
        # for key in data_dict:
        #     # if 'depth' in key:
        #         # print(data_dict[key])
        #     print(key, data_dict[key].shape, torch.min(data_dict[key]), torch.max(data_dict[key]))
        
        # keys = ['ref', 'image', 'depth', 'normal', 'obj']
        # alloutdict = {}
        # for key in keys:
        #     alloutdict[key] = []
        # alloutdict['ref'] = np.concatenate([temporal_ref_image_pt.numpy() for _ in range(self.frame_num)], axis=0)
        # alloutdict['image'] = temporal_camera_image_pt.numpy()
        # alloutdict['depth'] = temporal_depth_image_pt.numpy()
        # alloutdict['normal'] = temporal_normal_image_pt.numpy()
        # alloutdict['obj'] = temporal_obj_ref_middle_image_pt.numpy()
        # vid_name = os.path.join(debug_vid_dir, f'./debug_{frames_start}_vid.mp4')
        # save_dict_images_to_mp4(alloutdict, keys, vid_name, fps=1)

        return data_dict



class HuggingfaceDataset(BaseDataset):
    def __init__(self, dataset_name, **kwargs):
        super(HuggingfaceDataset , self).__init__(kwargs.get('config_path', None), dataset_name, kwargs.get('transform', None))

        # dataset = load_dataset('/shared_disk/users/wenkang.qin/data/robotransfer-training-data/robotransfer-dataset-training/')
        # dataset = dataset['train']

        # dataset = load_from_disk('/horizon-bucket/robot_lab/users/jiagang.zhu/dataset/results/add_obj_1000')

        # print("dataset_name: ", dataset_name)
        dataset = load_from_disk(dataset_name)

        self.dataset = dataset
        self.lendataset = len(dataset)
        self.frame_num = kwargs.get('frame_num', 30)
        
        self.idx_list = [idx for idx in range(0, len(dataset), self.frame_num)]
        
        data_shuffle = kwargs.get('data_shuffle', True)

        self.multiview = kwargs.get('multiview', True)
        print("multiview: ", self.multiview)
        print("frame_num: ", self.frame_num)

        model_path = "/horizon-bucket/robot_lab/users/jeff.wang/models/public/huggingface/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1"
        self.clip_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder='feature_extractor')
        
        self.target_size = kwargs.get('target_size', (192, 128))
        
        print("self.target_size: ", self.target_size)
        
        if data_shuffle:
            np.random.shuffle(self.idx_list)
            
    def __len__(self):
        return len(self.idx_list)

    def _get_data(self, index):
        
        data_dict = {}

        frames_start = self.idx_list[index]
        scene_id = self.dataset[frames_start]["scene_id"]

        temporal_depth_images = []
        temporal_normal_images = []
        temporal_camera_images = []
        
        colored_depth = True
        
        for i in range(frames_start, frames_start+self.frame_num):
            if i >= self.lendataset:
                i = self.lendataset - 1
            # padding if scene_id has changed
            if self.dataset[i]["scene_id"] != scene_id:
                temporal_depth_images.append(temporal_depth_images[-1])
                temporal_normal_images.append(temporal_normal_images[-1])
                temporal_camera_images.append(temporal_camera_images[-1])
                continue
            
            depth_left_image, depth_middle_image, depth_right_image = self.dataset[i][
                "depth_images"
            ]

            if self.target_size is not None:
                depth_left_image = depth_left_image.resize(size=self.target_size)
                depth_middle_image = depth_middle_image.resize(size=self.target_size)
                depth_right_image = depth_right_image.resize(size=self.target_size)

            normal_left_image, normal_middle_image, normal_right_image = self.dataset[i][
                "normal_images"
            ]

            if self.target_size is not None:
                normal_left_image = normal_left_image.resize(size=self.target_size)
                normal_middle_image = normal_middle_image.resize(size=self.target_size)
                normal_right_image = normal_right_image.resize(size=self.target_size)

            camera_left_image, camera_middle_image, camera_right_image = self.dataset[i][
                "camera_images"
            ]

            if self.target_size is not None:
                camera_left_image = camera_left_image.resize(size=self.target_size)
                camera_middle_image = camera_middle_image.resize(size=self.target_size)
                camera_right_image = camera_right_image.resize(size=self.target_size)

            if self.multiview:
                if colored_depth:
                    depth_image = np.concatenate(
                        [
                            np.array(depth_left_image) / 255,
                            np.array(depth_middle_image) / 255,
                            np.array(depth_right_image) / 255,
                        ],
                        axis=1,
                    )
                else:
                    depth_image = np.concatenate(
                        [
                            np.array(depth_left_image) / 1000,
                            np.array(depth_middle_image) / 1000,
                            np.array(depth_right_image) / 1000,
                        ],
                        axis=1,
                    )
                    
                    
                normal_image = np.concatenate(
                    [
                        np.array(normal_left_image) / 255,
                        np.array(normal_middle_image) / 255,
                        np.array(normal_right_image) / 255,
                    ],
                    axis=1,
                )
                camera_image = np.concatenate(
                    [
                        np.array(camera_left_image) / 255,
                        np.array(camera_middle_image) / 255,
                        np.array(camera_right_image) / 255,
                    ],
                    axis=1,
                )
            else:
                if colored_depth:
                    depth_image = np.array(depth_middle_image) / 255
                else:
                    depth_image = np.array(depth_middle_image) / 1000
                normal_image = np.array(normal_middle_image) / 255
                camera_image =  np.array(camera_middle_image) / 255
                
                
            temporal_depth_images.append(depth_image)
            temporal_normal_images.append(normal_image)
            temporal_camera_images.append(camera_image)

        temporal_obj_ref_middle_images = []
        temporal_obj_ref_left_images = []
        temporal_obj_ref_right_images = []
        
        object_reference_left_images = self.dataset[frames_start][
            "object_reference_left_images"
        ]
        object_reference_middle_images = self.dataset[frames_start][
            "object_reference_middle_images"
        ]
        object_reference_right_images = self.dataset[frames_start][
            "object_reference_right_images"
        ]
        num_objs = len(object_reference_middle_images)
        
        # import cv2
        
        MAX_NUM_OBJS = 10
        def resize_obj_list(images):
            new_images = []
            cnt = 0
            for _image in images:
                _image = _image.resize(size=self.target_size)
                _image = _image.convert('RGB')
                _image = np.array(_image)
                new_images.append(_image)
                cnt += 1
                if cnt == MAX_NUM_OBJS:
                    break
            while cnt < MAX_NUM_OBJS:
                _image0 = np.zeros((self.target_size[1], self.target_size[0], 3)).astype("uint8")
                new_images.append(_image0)
                cnt += 1
            return new_images
        
        object_reference_left_images = resize_obj_list(object_reference_left_images)
        object_reference_middle_images = resize_obj_list(object_reference_middle_images)
        object_reference_right_images = resize_obj_list(object_reference_right_images)
        
        temporal_ref_images = []
        reference_left_image, reference_middle_image, reference_right_image = self.dataset[frames_start][
            "reference_images"
        ]
        if self.target_size is not None:
            reference_left_image = reference_left_image.resize(size=self.target_size)
            reference_middle_image = reference_middle_image.resize(size=self.target_size)
            reference_right_image = reference_right_image.resize(size=self.target_size)

        if self.multiview:
            reference_image = np.concatenate(
                [
                    np.array(reference_left_image) / 255,
                    np.array(reference_middle_image) / 255,
                    np.array(reference_right_image) / 255,
                ],
                axis=1,
            )
        else:
            reference_image = np.array(reference_middle_image) / 255       
        
        temporal_ref_images.append(reference_image)
        temporal_depth_image = np.stack(temporal_depth_images, axis=0)
        
        if colored_depth:
            temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).permute(
                0, 3, 1, 2
            )   
        else:         
            temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(dim=1)
            temporal_depth_image_pt = temporal_depth_image_pt.clip(
                min=0.1, max=2.0
            )
            
        temporal_normal_image = np.stack(temporal_normal_images, axis=0)
        temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
            0, 3, 1, 2
        )
        temporal_normal_image_pt = 2 * temporal_normal_image_pt - 1

        temporal_camera_image = np.stack(temporal_camera_images, axis=0)
        temporal_camera_image_pt = torch.from_numpy(temporal_camera_image).permute(
            0, 3, 1, 2
        )
        temporal_camera_image_pt = 2 * temporal_camera_image_pt - 1

        temporal_ref_image = np.stack(temporal_ref_images, axis=0)
        temporal_ref_image_pt = torch.from_numpy(temporal_ref_image).permute(
            0, 3, 1, 2
        )
        temporal_ref_image_pt = 2 * temporal_ref_image_pt - 1

        temporal_obj_ref_middle_image = np.stack(object_reference_middle_images, axis=0) / 255
        temporal_obj_ref_middle_image_pt = torch.from_numpy(temporal_obj_ref_middle_image).permute(
            0, 3, 1, 2
        )
        temporal_obj_ref_middle_image_pt = 2 * temporal_obj_ref_middle_image_pt - 1
        
        data_dict = {}
        data_dict['concat'] = temporal_camera_image_pt
        data_dict['concat_depth'] = temporal_depth_image_pt
        data_dict['concat_normal'] = temporal_normal_image_pt
        data_dict['concat_ref_image'] = temporal_ref_image_pt
        data_dict['concat_ref_obj_image'] = temporal_obj_ref_middle_image_pt
        
        input_clip_image = _resize_with_antialiasing(data_dict['concat_ref_image'], (224, 224))
        input_clip_image = (input_clip_image + 1.0) / 2.0
        input_clip_image = self.clip_processor(
            images=input_clip_image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors='pt',
        ).pixel_values
        data_dict['concat_clip_image'] = input_clip_image

        input_clip_image = _resize_with_antialiasing(data_dict['concat_ref_obj_image'], (224, 224))
        input_clip_image = (input_clip_image + 1.0) / 2.0
        input_clip_image = self.clip_processor(
            images=input_clip_image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors='pt',
        ).pixel_values
        data_dict['concat_obj_clip_image'] = input_clip_image
        
        num_pad_objs = MAX_NUM_OBJS - num_objs
        encoder_attention_mask = torch.zeros((1, MAX_NUM_OBJS + 1), device=input_clip_image.device, dtype=input_clip_image.dtype)
        if num_pad_objs > 0:
            encoder_attention_mask[:, -num_pad_objs:] = float('-inf')
        data_dict['encoder_attention_mask'] = encoder_attention_mask
        
        # temporal_camera_image_pt = (temporal_camera_image_pt + 1.0)/2 * 255
        # temporal_normal_image_pt = (temporal_normal_image_pt + 1.0)/2 * 255
        # temporal_depth_image_pt = (temporal_depth_image_pt + 1.0)/2   * 255
        
        # temporal_camera_image_pt = temporal_camera_image_pt.permute(
        #     0, 2, 3, 1
        # )
        # temporal_normal_image_pt = temporal_normal_image_pt.permute(
        #     0, 2, 3, 1
        # )
        # temporal_depth_image_pt = temporal_depth_image_pt.permute(
        #     0, 2, 3, 1
        # )           
        # temporal_ref_image_pt = temporal_ref_image_pt.permute(
        #     0, 2, 3, 1
        # )           
        # output_path = './debug_temporal_camera_image_pt.mp4'
        # save_images_to_mp4(temporal_camera_image_pt.numpy(), output_path, fps=30)
        
        # output_path = './debug_temporal_normal_image_pt.mp4'
        # save_images_to_mp4(temporal_normal_image_pt.numpy(), output_path, fps=30)

        # output_path = './debug_temporal_depth_image_pt.mp4'
        # save_images_to_mp4(temporal_depth_image_pt.numpy(), output_path, fps=30)
       
        # output_path = './debug_temporal_ref_image_pt.mp4'
        # save_images_to_mp4(temporal_ref_image_pt.numpy(), output_path, fps=30)
        
        # for key in data_dict:
        #     if 'depth' in key:
        #         print(data_dict[key])
        #         print(key, data_dict[key].shape, torch.min(data_dict[key]), torch.mid(data_dict[key]), torch.max(data_dict[key]))

        return data_dict

    
import imageio
def save_images_to_mp4(images, output_path, fps=30):
    frames = [np.array(frame) for frame in images]
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
    
def save_dict_images_to_mp4(outdict, keys, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for key in keys:
        lent = len(outdict[key])
        break
    frames = []
    for i in range(lent):
        frame_i_list = []
        for key in keys:
            frame = outdict[key][i]
            frame_i_list.append(np.array(frame))
        frame = np.concatenate(frame_i_list, axis=1)
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")