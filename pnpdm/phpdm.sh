#!/bin/bash
#BSUB -J grid_search               # Job name
#BSUB -o /project/cigserver3/export/w.weining/pnpdm/pnpdm/log/dem_train.log                    # Standard output
#BSUB -e /project/cigserver3/export/w.weining/pnpdm/pnpdm/log/dem_train_error.log                    # Standard error
#BSUB -q gpu-compute                      # GPU queue
#BSUB -gpu "num=1:gmodel=NVIDIAA10080GBPCIe"
#BSUB -n 4                            # 4 CPU cores
#BSUB -W 168:00                          # 7 days  runtime

# Activate Python environment
source /project/cigserver3/export/w.weining/miniconda3/etc/profile.d/conda.sh
conda activate pnpdm

# Navigate to script location
cd /project/cigserver3/export/w.weining/pnpdm/pnpdm
python posterior_sample.py +data=ffhq_grayscale +task=IDT +model=edm_unet_adm_gray_ffhq +sampler=pnp_edm \
       sampler.mode=vp_sde sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 gpu=0 add_exp_name=anneal-0.9
#torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs     --data=/project/cigserver3/export/w.weining/Shirin_EDM/P+Q_half.zip --cond=0 --arch=ddpmpp --batch=2