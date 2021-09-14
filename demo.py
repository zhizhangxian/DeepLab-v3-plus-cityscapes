from cityscapes import CityScapes, collate_fn2

from tqdm import tqdm
from torch.utils.data import DataLoader
from configs.configurations import Config


if __name__ == '__main__':
    cfg = Config()
    cfg.datapth = r'D:\datasets\cityscapes'
    ds = CityScapes(cfg, mode='train', num_copys=2)
    # print(ds[0])

    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    collate_fn=collate_fn2,
                    drop_last = True)
    for im_lb in dl:
        print(im_lb)
        break

