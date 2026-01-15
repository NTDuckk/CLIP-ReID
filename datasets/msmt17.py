
# import glob
# import re

# import os.path as osp

# from .bases import BaseImageDataset


# class MSMT17(BaseImageDataset):
#     """
#     MSMT17

#     Reference:
#     Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

#     URL: http://www.pkuvmc.com/publications/msmt17.html

#     Dataset statistics:
#     # identities: 4101
#     # images: 32621 (train) + 11659 (query) + 82161 (gallery)
#     # cameras: 15
#     """
#     dataset_dir = 'MSMT17'

#     def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
#         super(MSMT17, self).__init__()
#         self.pid_begin = pid_begin
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.dataset_dir, 'train')
#         self.test_dir = osp.join(self.dataset_dir, 'test')
#         self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
#         self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
#         self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
#         self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

#         self._check_before_run()
#         train = self._process_dir(self.train_dir, self.list_train_path)
#         val = self._process_dir(self.train_dir, self.list_val_path)
#         train += val
#         query = self._process_dir(self.test_dir, self.list_query_path)
#         gallery = self._process_dir(self.test_dir, self.list_gallery_path)
#         if verbose:
#             print("=> MSMT17 loaded")
#             self.print_dataset_statistics(train, query, gallery)

#         self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError("'{}' is not available".format(self.train_dir))
#         if not osp.exists(self.test_dir):
#             raise RuntimeError("'{}' is not available".format(self.test_dir))

#     def _process_dir(self, dir_path, list_path):
#         with open(list_path, 'r') as txt:
#             lines = txt.readlines()
#         dataset = []
#         pid_container = set()
#         cam_container = set()
#         for img_idx, img_info in enumerate(lines):
#             img_path, pid = img_info.split(' ')
#             pid = int(pid)  # no need to relabel
#             camid = int(img_path.split('_')[2])
#             img_path = osp.join(dir_path, img_path)
#             dataset.append((img_path, self.pid_begin+pid, camid-1, 0))
#             pid_container.add(pid)
#             cam_container.add(camid)
#         print(cam_container, 'cam_container')
#         # check if pid starts from 0 and increments with 1
#         for idx, pid in enumerate(pid_container):
#             assert idx == pid, "See code comment for explanation"
#         return dataset

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    Folder-scan MSMT17 (Market1501-style package):
      MSMT17/
        train/   (was bounding_box_train)
        test/    (was bounding_box_test)   -> used as gallery
        query/
    Filenames like: 0000_c1_0000.jpg (pid=0, cam=1)
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.query_dir = osp.join(self.dataset_dir, 'query')

        self._check_before_run()

        train = self._process_dir_scan(self.train_dir)
        query = self._process_dir_scan(self.query_dir)
        gallery = self._process_dir_scan(self.test_dir)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _parse_pid_cam(self, fname: str):
        """
        Parse filenames like:
          0000_c1_0000.jpg  -> pid=0, cam=1
          0123_c12_0456.png -> pid=123, cam=12
        """
        base = osp.basename(fname)

        # pid: token before first '_'
        pid_str = base.split('_')[0]
        if pid_str in ['-1', '0000', '0'] and False:
            # NOTE: do NOT skip pid=0 for your dataset (MSMT17 has pid=0)
            pass
        try:
            pid = int(pid_str)
        except:
            return None, None

        # cam: look for c<number>
        m = re.search(r'c(\d+)', base)
        if not m:
            return None, None
        cam = int(m.group(1))

        return pid, cam

    def _process_dir_scan(self, dir_path):
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths += glob.glob(osp.join(dir_path, ext))
        img_paths = sorted(img_paths)

        dataset = []
        for img_path in img_paths:
            pid, camid = self._parse_pid_cam(img_path)
            if pid is None:
                continue
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 0))

        if len(dataset) == 0:
            raise RuntimeError(f"No images found in {dir_path}. Check dataset structure/filenames.")
        return dataset