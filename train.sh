# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
CUDA_VISIBLE_DEVICES=0 \
python main.py --base configs/latent-diffusion/wave-config.yaml -t \
--gpus 0,
# --gpus 0,1,2,3,4,5,6