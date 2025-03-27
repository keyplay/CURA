CUDA_VISIBLE_DEVICES=1,2,3 python main_cross_domains.py --num_runs 5 --dataset CMAPSS --da_method Deep_Coral & 
CUDA_VISIBLE_DEVICES=1,2,3 python main_cross_domains.py --num_runs 5 --dataset CMAPSS --da_method DDC &
CUDA_VISIBLE_DEVICES=1,2,3 python main_cross_domains.py --num_runs 5 --dataset CMAPSS --da_method DANN_cons &
CUDA_VISIBLE_DEVICES=1,2,3 python main_cross_domains.py --num_runs 5 --dataset CMAPSS --da_method CADA &
CUDA_VISIBLE_DEVICES=1,2,3 python main_cross_domains.py --num_runs 5 --dataset CMAPSS --da_method ADARUL &
wait
:
