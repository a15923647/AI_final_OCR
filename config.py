from PIL import ImageFilter
from PIL import ImageEnhance
import pprint
original_preprocess = lambda x: x
gaussianBlur_preprocess = lambda x: x.filter(ImageFilter.GaussianBlur(1))
enhance_preprocess = lambda x: ImageEnhance.Contrast(x).enhance(2)
emboss_preprocess = lambda x: x.filter(ImageFilter.EMBOSS)
common_config = {
    'data_dir': 'data/mnt/ramdisk/max/90kDICT32px/', # The root directory of Synth90k dataset, which contains multiple annotation files.
    'img_width': 100, # Adjust size of all images before fed into CRNN model.
    'img_height': 32,
    'map_to_seq_hidden': 64, # Input channel of RNN model.
    'rnn_hidden': 256, # Size of hidden level of RNN.
    'leaky_relu': False, # Use leaky_relu in CNN; otherwise, use relu instead.
    'img_limit': 10000, # The dataset size.
    'use_mem_buffer': True, # Allow Dataset to cache all images in memory instead of reading images from disks repeatly.
    'start_sample_index': 10000, # The Dataset will contain images from the start_sample_index to start_sample_index-1 image from original Synth90k dataset.
    'image_preprocess': emboss_preprocess, # The preprocessing function before output to the model. #here
    'dump_dest': "E:\\synth90k", # If use_mem_buffer, the dataset will automatically load dataset cache from dumped pickle file to accelerate IO.
    'dump_name_suffix': '_emboss', # The suffix of output pickle file. #here
    'save_pkl': True # Save the cache to a pickle file upon the entire dataset is loaded.
}
entire = eval_res_mid_name = "emboss" # Content to identify different evaluation result json file. #here
test_dir_config = {
    'eval_res_name': f"{common_config['img_limit']}_{eval_res_mid_name}_eval_res.json",
    'lookup_dirs' : ['checkpoints_original/',# Test files matching ^crnn_[0-9]{6}.*pt$ format in these folders.
                     'checkpoints_enhance2/', 
                     'checkpoints_gaussian_100000/',
                     'checkpoints_emboss/',
                     'checkpoints_org_200000/', 
                     'checkpoints_org_300000/', 
                     'checkpoints_org_1000000/',  
                     'checkpoints_gaussian_200000/', 
                     'checkpoints_enhance_200000/']

}
# training settings
train_config = {
    'epochs': 10000,
    'train_batch_size': 32,
    'eval_batch_size': 256,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 0,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)
# evaluation settings
evaluate_config = {
    'eval_batch_size': 256,
    'cpu_workers': 0,
    'reload_checkpoint': 'checkpoints/crnn_synth90k.pt',
    'decode_method': 'greedy',
    'beam_size': 10,
}
evaluate_config.update(common_config)
