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
| NeurlPS 2022 Dataset and Benchmark | [MBW: Multi-view Bootstrapping in the Wild](https://arxiv.org/pdf/2210.01721) | Raw Dataset | Multi-veiw (2~4 cameras) tigers, fish, colobus monkeys, gorillas, chimpanzees, and flamingos from a zoo dataset,each with 2 synchronized videos | Multi-view, 2D landmark of articulated objects | Labeling articulated objects |
| CVPR 2023 | [OMNI3D: A Large Benchmark and Model for 3D Object Detection in the Wild](https://arxiv.org/pdf/2207.10660) | New annotation (SUN RBG-D, ARKitScenes, Hypersim, Objectron, KITTI and nuScenes) | 234k images annotated with more than 3 million instances and 98 3D boxes categories | single-image, 3D cuboids | 3D object detections |
| CVPR 2022 | [Self-supervised Neural Articulated Shape and Appearance Models](https://arxiv.org/pdf/2205.08525) | No dataset contribution | - | image, 3D shape | few-shot reconstruction, the generation of novel articulations, and novel view-synthesis |

#### Hands
- (NeurlPS 2023) **A Dataset of Relighted 3D Interacting Hands** [two-hand interactive], [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/396beafa6feba781a7114780e6837253-Paper-Datasets_and_Benchmarks.pdf)


### Scene

| time | paper | Sources| Data Scale| Modality | Task | Model |
|----------|----------|----------|----------|----------|----------|----------|
| CVPR 2024 | [LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset](https://arxiv.org/pdf/2312.12418) | ArKitScenes | 10,412 CAD aligned with 920 scenes across 17 categories scaned from ArKitScene| Point cloud, Multi-view | indoor instance-level scene reconstruction | Diffusion-based |
| | []() | | | | |


## Real-Synethetic
- (CVPR 2024) **SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D Scenes** [[Paper]](https://alexdelitzas.github.io/assets/pdf/SceneFun3D.pdf)
- (ECCV 2022) **OPD: Single-view 3D Openable Part Detection** [Openable Part Detection],[[Paper]](https://arxiv.org/pdf/2203.16421)


## Synthetic


### Object
| time | paper | Sources| Data Scale| Modality | Task |
|----------|----------|----------|----------|----------|----------|
| CVPR 2024| [Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering](https://arxiv.org/pdf/2312.11360)| 
| CVPR 2023 | [GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts](https://arxiv.org/pdf/2211.05272) | Raw: GAPartNet| 8489part instances on 1166 objects  | Point-cloud | part segmentation, part pose estimation, and part-based object manipulation |
| ArXiv 2212 | [GeoCode: Interpretable Shape Programs](https://arxiv.org/pdf/2212.11715) | - | train: 9,570 chairs, 9,330 vases, and 6,270 tables; validation and test: 957 chairs, 933 vases, and 627 tables | Mesh, Point-Cloud, sketch | 3D geometry edit |
| NeurlPS 2022 Datasets and Benchmarks | [Breaking Bad: A Dataset for Geometric Fracture and Reassembly](https://breaking-bad-dataset.github.io) | Thingi10K, PartNet | 10,474 shapes, 1,047,400 breakdown patterns | Point cloud | geometry measurements; shape assembly |
| CVPR 2022 | [Fixing Malfunctional Objects With Learned Physical Simulation and Functional Prediction](https://arxiv.org/pdf/2205.02834) | Raw Data: poorly-designed 3D physical objects (point videos of 3D objects) with choices to fix them | 5K | Point cloud | fixing 3D object shapes based on functionality |
| | []() | | | | |

#### Manipulation
| time | paper | Sources| Data Scale| Modality | Task |
|----------|----------|----------|----------|----------|----------|
| CoRL 2022 | [Leveraging Language for Accelerated Learning of Tool Manipulation](https://arxiv.org/pdf/2206.13074) | - | 36 objects | images | tool utilize |



### Object-Scene

| time | paper | Sources| Data Scale| Modality | Task |
|----------|----------|----------|----------|----------|----------|
| NeurlPS 2022 | [PeRFception: Perception using Radiance Fields](https://openreview.net/pdf?id=MzaPEKHv-0J) | CO3D, ScanNet | Co3D(18669 annotated videos with a total 1.5 million of camera-annotated frames), ScanNet(1.5 K indoor scenes with commercial RGB-D sensors) | Multi-view, reconstructed Point-cloud | 2D image classification, 3D object classification, 3D semantic segmentation |



## Generative

### Scene
- (2022.10) **CommonSim-1: Generating 3D Worlds** [[Project]](https://www.csm.ai/blog/commonsim-1-generating-3d-worlds), text-to-3D dynamic environment.

### Object
- (CVPR 2024) **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ling_Align_Your_Gaussians_Text-to-4D_with_Dynamic_3D_Gaussians_and_Composed_CVPR_2024_paper.pdf)
- (CVPR 2024) **Wonder3D: Single Image to 3D using Cross-Domain Diffusion** [[Paper]](https://arxiv.org/pdf/2310.15008)
- (NeurlPS 2024) **One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization** [[Paper]](https://arxiv.org/pdf/2306.16928)
- (2022.12) **Point·E: A System for Generating 3D Point Clouds from Complex Prompts** [[Paper]](https://arxiv.org/pdf/2212.08751) 
- (ICLR 2024) **SyncDreamer: Generating Multiview-consistent Images from a Single-view Image** [[Paper]](https://arxiv.org/abs/2309.03453)