#!/bin/bash -x

python evaluate.py --result_path ./results/pix2pixL1

python evaluate.py --result_path ./results/pix2pixL1S1

python evaluate.py --result_path ./results/pix2pixL1S1S2

python evaluate.py --result_path ./results/pix2pixL1S1S2S3

python evaluate.py --result_path ./results/pix2pixfft

python evaluate.py --result_path ./results/pix2pixfft_only

python evaluate.py --result_path ./results/pix2pixfft_1
