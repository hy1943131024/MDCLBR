# MDCLBR: Multiview Denoising Contrastive Learning  for Bundle Recommendation
## How to run the code
1. Decompress the dataset file into the current folder: 

   > tar -zxvf dataset.tgz

2. Train MDCLBR on the dataset Youshu with GPU 0: 

   > python train.py -g 0 -m MDCLBR -d Youshu

   You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file. The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
