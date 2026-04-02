import os
import re
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 两个图片目录（直接放图片，不含子文件夹逻辑）
DIR_BASELINE = "/home/whut/Code/G-FSDet/G-FSDet-main/work_dirs/dior/rep/tfa_r101_fpn_dior_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning_/show-dir_gfsdet_ori_dior/DIOR2017/JPEGImages"
DIR_CKT = "/home/whut/Code/G-FSDet/G-FSDet-main/work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning/ContrastiveAttentionFusion/show-dir_CKT_final_dior/DIOR2017/JPEGImages"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def natural_key(s):
    """
    自然排序：
    例如 1.jpg, 2.jpg, 10.jpg
    不会排成 1,10,2
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def collect_pairs(dir_baseline, dir_ckt):
    """
    收集两个目录下的同名图片
    返回:
        [
            {
                "name": "1.jpg",
                "baseline_path": "...",
                "ckt_path": "..."
            },
            ...
        ]
    """
    if not os.path.isdir(dir_baseline):
        raise ValueError(f"Baseline 路径不存在: {dir_baseline}")
    if not os.path.isdir(dir_ckt):
        raise ValueError(f"CKT 路径不存在: {dir_ckt}")

    baseline_imgs = sorted(
        [f for f in os.listdir(dir_baseline) if is_image_file(f)],
        key=natural_key
    )
    ckt_imgs = sorted(
        [f for f in os.listdir(dir_ckt) if is_image_file(f)],
        key=natural_key
    )

    common_imgs = sorted(set(baseline_imgs) & set(ckt_imgs), key=natural_key)

    pairs = []
    for img_name in common_imgs:
        pairs.append({
            "name": img_name,
            "baseline_path": os.path.join(dir_baseline, img_name),
            "ckt_path": os.path.join(dir_ckt, img_name)
        })

    return pairs


class ImagePairViewer:
    def __init__(self, pairs):
        if not pairs:
            raise ValueError("没有找到可对比的同名图片。")

        self.pairs = pairs
        self.idx = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.18, top=0.9, wspace=0.05)

        # 按钮区域
        ax_prev = plt.axes([0.32, 0.05, 0.12, 0.05])
        ax_next = plt.axes([0.56, 0.05, 0.12, 0.05])

        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def current_image_data(self):
        return self.pairs[self.idx]

    def update_display(self):
        image_data = self.current_image_data()

        img1 = self.read_image(image_data["baseline_path"])
        img2 = self.read_image(image_data["ckt_path"])

        self.ax1.clear()
        self.ax2.clear()

        if img1 is not None:
            self.ax1.imshow(img1)
        self.ax1.set_title(f"Baseline\n{image_data['name']}", fontsize=12)
        self.ax1.axis("off")

        if img2 is not None:
            self.ax2.imshow(img2)
        self.ax2.set_title(f"CKT\n{image_data['name']}", fontsize=12)
        self.ax2.axis("off")

        self.fig.suptitle(
            f"Image: {self.idx + 1}/{len(self.pairs)}    Filename: {image_data['name']}",
            fontsize=14
        )

        self.fig.canvas.draw_idle()

    def next_image(self, event=None):
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
            self.update_display()

    def prev_image(self, event=None):
        if self.idx > 0:
            self.idx -= 1
            self.update_display()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.next_image()
        elif event.key in ['left', 'a']:
            self.prev_image()

    def show(self):
        plt.show()


if __name__ == "__main__":
    pairs = collect_pairs(DIR_BASELINE, DIR_CKT)
    print(f"[INFO] 找到 {len(pairs)} 对可对比图片")
    viewer = ImagePairViewer(pairs)
    viewer.show()