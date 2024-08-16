# Awesome-3D-Dataset

## Real Data

### Object

| time | paper | Sources| Data Scale| Modality | Task |
|----------|----------|----------|----------|----------|----------|
| NeurlPS 2023| [OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding](https://arxiv.org/pdf/2305.10764) | ShapeNetCore, 3D-FUTURE,ABO, Objaverse| 876 K | Text-image-3D point cloud | point cloud captioning, point-cloud conditined image generation |
| CVPR 2024 | [ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding](https://arxiv.org/pdf/2305.08275) | Objaverse, ShapeNet | (800K real-world 3D shape),(52.5K 3D shapes with 55 annotated categories) | 3D point clouds, images, and language | zero-shot 3D classification, standard 3D classification with fine-tuning, and 3D captioning (3D-to- language generation) |
| CVPR2023 | [RealImpact: A Dataset of Impact Sound Fields for Real Objects](https://arxiv.org/pdf/2306.09944) | Raw | 150,000 recordings of impact sounds of 50 everyday objects,5 distinct impact positions | impact locations, microphone locations, contact force profiles, material labels, and RGBD images | listener location classification and visual acoustic matching |
| Workshop | [3DCoMPaT Challenge](https://3dcompat-dataset.org/workshop/C3DV24/) | 3DCoMPaT dataset++ |  | 3D objects, 3D renderings |  recognize and ground compositions of materials on parts of 3D objects | 
| CVPR 2024 | [LASO: Language-guided Affordance Segmentation on 3D Object](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.pdf) | 3D-AffordanceNet | 19,751 point-question pairs, covering 8434 object shapes and 870 expert-crafted questions; | Point Cloud, Text | Language-guided Affordance Segmentation on 3D Object |

### Scene

| time | paper | Sources| Data Scale| Modality | Task | Model |
|----------|----------|----------|----------|----------|----------|----------|
| CVPR 2024 | [LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset](https://arxiv.org/pdf/2312.12418) | ArKitScenes | 10,412 CAD aligned with 920 scenes across 17 categories scaned from ArKitScene| Point cloud, Multi-view | indoor instance-level scene reconstruction | Diffusion-based |


### Synthetic


### Object
| time | paper | Sources| Data Scale| Modality | Task |
|----------|----------|----------|----------|----------|----------|
| CVPR 2024| [Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering](https://arxiv.org/pdf/2312.11360)| 
| CVPR 2023 | [GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts](https://arxiv.org/pdf/2211.05272) | Raw: GAPartNet| 8489part instances on 1166 objects  | Point-cloud | part segmentation, part pose estimation, and part-based object manipulation |
| | []() | | | | |
