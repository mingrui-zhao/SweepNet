<p align="center">

  <h1 align="center"><a href="https://mingrui-zhao.github.io/SweepNet/" target="_blank">SweepNet: Unsupervised Learning Shape Abstraction via Neural Sweepers</a></h1>

  <p align="center">
    <a href="https://sairajk.github.io/" target="_blank"><strong>Mingrui Zhao</strong></a>
    ·
    <a href="https://yizhiwang96.github.io/" target="_blank"><strong>Yizhi Wang</strong></a>
    ·
    <a href="https://fenggenyu.github.io/" target="_blank"><strong>Fenggen Yu</strong></a>
    ·
    <a href="https://changqingzou.weebly.com/" target="_blank"><strong>Changqing Zou</strong></a>
    ·
    <a href="https://arash-mham.github.io/" target="_blank"><strong>Ali Mahdavi-Amiri</strong></a>
    <br />
    <i>European Conference on Computer Vision (ECCV), 2024</i>    
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2407.06305" target="_blank"><strong>arXiv</strong></a>
    |
    <a href="https://mingrui-zhao.github.io/SweepNet/" target="_blank"><strong>Project Page</strong></a>
  </p>

  <div  align="center">
    <video width="80%" playsinline="" autoplay="autoplay" loop="loop" preload="" muted="">
        <source src="asset/SweepNet_greeting_1080.mp4" type="video/mp4">
    </video>
  </div>
</p>

This is the official implementation for SweepNet.
## Environment Setup
We recommend using Anaconda for managing the environment. Execute the following commands to create and activate the SweepNet environment:

```
conda create --name sweepnet python=3.8
conda activate sweepnet
pip install -r requirements.txt
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

## Dataset 
Download the preprocessed dataset GC-objects and quadrupeds at [huggingface](https://huggingface.co/datasets/zmrr/SweepNetDataset) or [Onedrive](https://1sfu-my.sharepoint.com/:f:/g/personal/mza143_sfu_ca/ElK2VX9kEeVNrCbJo4K-NKgBoLeVuOM8wHVKhERq3VwmDw?e=7bUexl).

Unzip the files and put them at **./data**

The GC-objects models are sourced from [OreX](https://arxiv.org/abs/2211.12886), [GCD](https://www.cs.sfu.ca/~haoz/pubs/zhou_siga15_gcd.pdf), and various internet sources. The quadruped dataset is adopted from [this paper](https://arxiv.org/abs/1612.00404).

Please cite these sources if you use our processed dataset.

## Neural Sweeper
Download the pretrained checkpoint of the neural sweeper [here](https://1sfu-my.sharepoint.com/:u:/g/personal/mza143_sfu_ca/EeXpzdgOOjtBoEIJZ8BkHqABTn0v3ZBDEIRGqjyB4oay7w?e=m2Qj32) and place the checkpoint in the **./neural_sweeper/ckpt** directory.

The training data is available on [huggingface](https://huggingface.co/datasets/zmrr/SweepNetDataset) and [Onedrive](https://1sfu-my.sharepoint.com/:f:/g/personal/mza143_sfu_ca/ElK2VX9kEeVNrCbJo4K-NKgBoLeVuOM8wHVKhERq3VwmDw?e=7bUexl). Please refer to our [paper](https://arxiv.org/abs/2407.06305) for data preparation details.


## Train the Model
### Voxel input
Execute the following command to train the SweepNet on a single shape with voxel input. The outcomes will be stored in the **./results** directory.

```python train.py --config_path ./configs/default.json```

### Pointcloud input
Execute the following command to train the SweepNet on a single shape with point cloud input. The outcomes will be stored in the **./results** directory.

```python train.py --config_path ./configs/pcd.json```

Execute the following command to train the SweepNet with different numbers of available primitives over the GC-object dataset. The outcomes will be stored in the **./results** directory.

```bash script/ablate_num_prim.sh```

Configuration options, including the target shape name, can be adjusted within the **./configs/default.json** file.

## Produce sweep surfaces from primitive parameters
Execute the command below to reproduce sweep surfaces from their parameters.

```python misc/produce_model_from_parameters.py```

Try some other primitive parameters we provided on the project page for fun! 😄

## Limitations
SweepNet prefers models exhibiting sweep elements and may converge to local optima with different initializations. If the abstraction result is unsatisfactory, try providing multiple runs for a better fit:

```bash script/train_multiple_runs.sh```

## Acknowledgements
Our codebased is developed based on [ExtrudeNet](https://github.com/kimren227/ExtrudeNet), [POCO](https://github.com/valeoai/POCO) and [CAPRI-NET](https://github.com/FENGGENYU/CAPRI-Net). The data pre-processing code is available at [D2CSG](https://github.com/FENGGENYU/D2CSG/tree/main/data_processing).

## Related Works
If you are interested in sweeping-based shape representation. Please also check out [SECAD-Net](https://arxiv.org/abs/2303.10613) with neural profiles and [ExtrudeNet](https://arxiv.org/abs/2209.15632) with Bezier polygon profiles.

## Citation
If you find our work interesing, please cite

```bibtex
@inproceedings{zhao2024sweepnet,
  title={SweepNet: Unsupervised Learning Shape Abstraction via Neural Sweepers},
  author={Zhao, Mingrui and Wang, Yizhi and Yu, Fenggen and Zou, Changqing and Mahdavi-Amiri, Ali},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  organization={Springer}
}
```
