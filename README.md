# *RoboTransfer*: Geometry-Consistent Video Diffusion for Robotic Visual Policy Transfer
[![🌐 Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://horizonrobotics.github.io/robot_lab/robotransfer)
[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2505.23171)
[![🎥 Video](https://img.shields.io/badge/🎥-Video-red)](https://youtu.be/dGXKtqDnm5Q)
[![中文介绍](https://img.shields.io/badge/中文介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/c9-1HPBMHIy4oEwyKnsT7Q)
[![机器之心介绍](https://img.shields.io/badge/机器之心介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/Hj2h3nxO8XxPeqd3OhctKA)

> ***RoboTransfer***, a diffusion-based video generation framework for robotic data synthesis. Unlike previous methods, RoboTransfer integrates multi-view geometry with explicit control over scene components, such as background and object attributes. By incorporating cross-view feature interactions and global depth/normal conditions, RoboTransfer ensures geometry consistency across views. This framework allows fine-grained control, including background edits and object swaps.
<img src="/assets/pin/robotransfer.jpg" alt="Overall Framework" width="700"/>

## ✅ Setup Environment

We use uv to manage dependencies, to get our environments:
```bash
git clone https://github.com/HorizonRobotics/RoboTransfer.git
cd RoboTransfer
export UV_HTTP_TIMEOUT=600
uv sync
uv pip install -e .
```

## 🚀 Inference

```bash
uv run main.py # --mem_efficient for 4090
```


## 📈 More Inference Data

Update the dependencies of the data pipeline.
```bash
uv sync --extra data
```

### ⚙️ For more sim data

You can obtain more simulation data from the [RoboTwin CVPR Challenge](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1).
 <!-- Alternatively, you can use the collected data in [RoboTwin2.0-aloha-agilex](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/aloha-agilex). -->
You can then use the process_sim.sh script to convert raw data (.pickle files and .hdf5) into the RoboTransfer format with geometric conditioning.

```bash
script/process_sim.sh
```

### 🤖 For more real data
For real-world data collected by the ALOHA-AgileX robot system, access the dataset [RoboTransfer-RealData](https://huggingface.co/datasets/HorizonRobotics/RoboTransfer-RealData). You can then process raw RGB images using the process_real.sh script to convert them into RoboTransfer format with geometric conditioning.

```bash
script/process_real.sh
```


## 🙌 Acknowledgement

RoboTransfer builds upon the following amazing projects and models:
🌟 [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
🌟 [Lotus](https://github.com/EnVision-Research/Lotus)
🌟 [GPT4o](https://platform.openai.com/docs/models/gpt-4o)
🌟 [GroundSam](https://github.com/IDEA-Research/Grounded-Segment-Anything)
🌟 [IOPaint](https://github.com/Sanster/IOPaint)

##  ⚖️ License
This project is licensed under the [Apache License 2.0](LICENSE). See the `LICENSE` file for details.

## 📚 Citation
If you use RoboTransfer in your research or projects, please cite:

```bibtex
@misc{liu2025robotransfergeometryconsistentvideodiffusion,
      title={RoboTransfer: Geometry-Consistent Video Diffusion for Robotic Visual Policy Transfer},
      author={Liu Liu and Xiaofeng Wang and Guosheng Zhao and Keyu Li and Wenkang Qin and Jiaxiong Qiu and Zheng Zhu and Guan Huang and Zhizhong Su},
      year={2025},
      eprint={2505.23171},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23171},
}
```
