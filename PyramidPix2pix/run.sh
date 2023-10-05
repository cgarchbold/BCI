
# Train L1 (Pix2pix)
python train.py --dataroot ./datasets/BCI --name pix2pixL1 --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1

# Train L1+S1
python train.py --dataroot ./datasets/BCI --name pix2pixL1S1  --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1_L2

# Train L1+S1+S2
python train.py --dataroot ./datasets/BCI --name pix2pixL1S1S2 --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1_L2_L3

# Train L1+S1+S2+S3
python train.py --dataroot ./datasets/BCI --name pix2pixL1S1S2S3 --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1_L2_L3_L4

# Train L1 (Pix2pix) UNet
python train.py --dataroot ./datasets/BCI --name pix2pixL1Unet --netG unet_256 --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1

python train.py --dataroot ./datasets/BCI --name pix2pixL1_test --preprocess crop --crop_size 512 --gpu_ids 0 --pattern L1

python train.py --dataroot ./datasets/BCI --name pix2pixfft --preprocess crop --crop_size 512 --gpu_ids 0 --pattern fft_L1

python train.py --dataroot ./datasets/BCI --name pix2pixfft_1 --preprocess crop --crop_size 512 --gpu_ids 0 --pattern fft

python -m visdom.server

