from pprint import pprint

# Use relative imports
from . import world, dataloader, model, utils
from . import rlmrec_models

import os
# Get datasets path relative to project root
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
_DATASETS_PATH = os.path.join(_PROJECT_ROOT, 'datasets')

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'amazon-book_subset_1500', 'amazon-book_subset_10000', 'amazon-book-2023']:
    dataset = dataloader.Loader(path=os.path.join(_DATASETS_PATH, world.dataset))
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'lgn_plus': rlmrec_models.LightGCN_plus,
    'lgn_gene': rlmrec_models.LightGCN_gene
}