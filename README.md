# MDCLBR: Multiview Denoising Contrastive Learning  for Bundle Recommendation
## How to run the code
1. Decompress the dataset file into the current folder: 

   > tar -zxvf dataset.tgz
 
   Noted: for the iFashion dataset, we incorporate three additional files: user\_id\_map.json, item\_id\_map.json, and bundle\_id\_map.json, which record the id mappings between the original string-formatted id in the POG dataset and the integer-formatted id in our dataset. You may use the mappings to obtain the original content information of the items/outfits. We do not use any content information in this work.

2. Train MDCLBR on the dataset Youshu with GPU 0: 

   > python train.py -g 0 -m MDCLBR -d Youshu

   You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file. The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
