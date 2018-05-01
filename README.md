# naturaladversary

1. Get the pretrained models for autoencoder, generator and discriminator 
https://drive.google.com/file/d/1E1Q5FHf1mUZsz7gPCUDNPsN-ZHKDxaNS/view?usp=sharing

2. export DATA_PATH = <path_to_above_data_folder_from_step1>

3. To train inverter, 
python3 train.py --data_path <path_to_data_folder> --update_base --convolution_enc -classifier_path <path_to_data_folder>/classifier
