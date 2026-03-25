import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 两个根目录
DIR_BASELINE = "/home/whut/Code/G-FSDet/G-FSDet-main/save_dir_baseline"
DIR_CKT = "/home/whut/Code/G-FSDet/G-FSDet-main/save_dir_CKT"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def collect_pairs(dir_baseline, dir_ckt):
    """
    收集两个根目录下，同名子文件夹中的同名图片
    返回:
        data = [
            {
                "folder": "000001",
                "images": [
                    {
                        "name": "P2_overlay.jpg",
                        "baseline_path": "...",
                        "ckt_path": "..."
                    },
                    ...
                ]
            },
            ...
        ]
    """
    baseline_subdirs = sorted([
        d for d in os.listdir(dir_baseline)
        if os.path.isdir(os.path.join(dir_baseline, d))
    ])
    ckt_subdirs = sorted([
        d for d in os.listdir(dir_ckt)
        if os.path.isdir(os.path.join(dir_ckt, d))
    ])

    common_subdirs = sorted(set(baseline_subdirs) & set(ckt_subdirs))
    data = []

    for sub in common_subdirs:
        baseline_subdir = os.path.join(dir_baseline, sub)
        ckt_subdir = os.path.join(dir_ckt, sub)

        baseline_imgs = sorted([
            f for f in os.listdir(baseline_subdir)
            if is_image_file(f)
        ])
        ckt_imgs = sorted([
            f for f in os.listdir(ckt_subdir)
            if is_image_file(f)
        ])

        common_imgs = sorted(set(baseline_imgs) & set(ckt_imgs))
        if not common_imgs:
            continue

        images = []
        for img_name in common_imgs:
            images.append({
                "name": img_name,
                "baseline_path": os.path.join(baseline_subdir, img_name),
                "ckt_path": os.path.join(ckt_subdir, img_name)
            })

        data.append({
            "folder": sub,
            "images": images
        })

    return data


class ImagePairViewer:
    def __init__(self, data):
        if not data:
            raise ValueError("没有找到可对比的同名子文件夹和同名图片。")

        self.data = data
        self.folder_idx = 0
        self.image_idx = 0

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.18, top=0.9, wspace=0.05)

        # 按钮区域
        ax_prev_img = plt.axes([0.20, 0.05, 0.10, 0.05])
        ax_next_img = plt.axes([0.32, 0.05, 0.10, 0.05])
        ax_prev_folder = plt.axes([0.50, 0.05, 0.12, 0.05])
        ax_next_folder = plt.axes([0.64, 0.05, 0.12, 0.05])

        self.btn_prev_img = Button(ax_prev_img, 'Prev Img')
        self.btn_next_img = Button(ax_next_img, 'Next Img')
        self.btn_prev_folder = Button(ax_prev_folder, 'Prev Folder')
        self.btn_next_folder = Button(ax_next_folder, 'Next Folder')

        self.btn_prev_img.on_clicked(self.prev_image)
        self.btn_next_img.on_clicked(self.next_image)
        self.btn_prev_folder.on_clicked(self.prev_folder)
        self.btn_next_folder.on_clicked(self.next_folder)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def current_folder_data(self):
        return self.data[self.folder_idx]

    def current_image_data(self):
        folder_data = self.current_folder_data()
        return folder_data["images"][self.image_idx]

    def update_display(self):
        folder_data = self.current_folder_data()
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
            f"Folder: {folder_data['folder']}   "
            f"Image: {self.image_idx + 1}/{len(folder_data['images'])}   "
            f"Folder Progress: {self.folder_idx + 1}/{len(self.data)}",
            fontsize=14
        )

        self.fig.canvas.draw_idle()

    def next_image(self, event=None):
        folder_data = self.current_folder_data()
        if self.image_idx < len(folder_data["images"]) - 1:
            self.image_idx += 1
        else:
            # 当前子文件夹看完后自动跳到下一个子文件夹第一张
            if self.folder_idx < len(self.data) - 1:
                self.folder_idx += 1
                self.image_idx = 0
        self.update_display()

    def prev_image(self, event=None):
        if self.image_idx > 0:
            self.image_idx -= 1
        else:
            if self.folder_idx > 0:
                self.folder_idx -= 1
                self.image_idx = len(self.current_folder_data()["images"]) - 1
        self.update_display()

    def next_folder(self, event=None):
        if self.folder_idx < len(self.data) - 1:
            self.folder_idx += 1
            self.image_idx = 0
        self.update_display()

    def prev_folder(self, event=None):
        if self.folder_idx > 0:
            self.folder_idx -= 1
            self.image_idx = 0
        self.update_display()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.next_image()
        elif event.key in ['left', 'a']:
            self.prev_image()
        elif event.key in ['down', 's']:
            self.next_folder()
        elif event.key in ['up', 'w']:
            self.prev_folder()

    def show(self):
        plt.show()


if __name__ == "__main__":
    data = collect_pairs(DIR_BASELINE, DIR_CKT)
    print(f"[INFO] 找到 {len(data)} 个可对比子文件夹")
    viewer = ImagePairViewer(data)
    viewer.show()