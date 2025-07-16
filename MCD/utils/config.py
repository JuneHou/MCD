# ----------------------running settings-------------------------- #
cp_data     = True      # using vqa-cp or not
version     = 'v2'      # 'v1' or 'v2'
train_set   = 'train'   # 'train' or 'train+val'
loss_type   = 'ce_margin'     # 'bce' or'ce'
in_memory   = False     # load all the image feature in memory

# ----------------------running settings-------------------------- #
entropy = 4.5
scale = 16
alpha = 0.5
temp = 0.2
use_cos = True
sc_epoch = 60
bias_inject = True
learnable_margins = FutureWarning
randomization = True
supcon = True
css = True

# ----------------------before-process data paths---------------- #
main_path       = '/data/wang/junh/datasets/vqa-cp-v2/'
qa_path         = main_path
bottom_up_path  = main_path + '/trainval_36/'  # raw image features
glove_path      = main_path + '/glove/glove.6B.300d.txt'

# ----------------------image id related paths------------------- #
ids_path    = '/data/wang/junh/githubs/MCD/MCD/AdaVQA/data'
image_path  = main_path + '/mscoco/' # image paths

# ----------------------processed data paths--------------------- #
rcnn_path           = main_path + '/rcnn-data/'
if cp_data:
    if version == 'v2':
        cache_root          = '/data/wang/junh/datasets/vqa-cp-v2/cp-cache/'
    else:
        cache_root = '/data/wang/junh/datasets/vqa-cp-v2/cpv1-cache/'
elif version == 'ok':
    cache_root = '/data/wang/junh/datasets/vqa-cp-v2/ok-cache/'
else:
    cache_root          = '/data/wang/junh/datasets/vqa-cp-v2/cache/'
dict_path           = qa_path + '/dictionary.json'
glove_embed_path    = '/data/wang/junh/datasets/vqa-cp-v2/glove6b_init_300d.npy'


# ----------------------running settings------------------------- #
max_question_len    = 14
image_dataset       = 'mscoco'
task                = 'OpenEnded' if not cp_data else 'vqacp'
test_split          = 'test2015' if not cp_data else 'test'   # 'test-dev2015' or 'test2015'
min_occurence       = 9             # answer frequency less than min will be omitted

# ----------------------preprocess image config------------------ #
num_fixed_boxes         = 36        # max number of object proposals per image
output_features         = 2048      # number of features in each object proposal
trainval_num_images     = 123287    # number of images for train and val
test_num_images         = 82783     # number of images for testing
