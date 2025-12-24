import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import roberts
from skimage.feature import canny
from skimage.morphology import skeletonize

# Tasvirlar
images = ["1.4.jpg"]

for img_path in images:
    # ===============================
    # 1-BOSQICH: ORIGINAL
    # ===============================
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_norm = gray / 255.0

    # ===============================
    # 2-BOSQICH: KONTUR AJRATISH
    # ===============================

    # Roberts
    roberts_edges = roberts(gray_norm)
    roberts_bin = roberts_edges > 0.05

    # Canny
    canny_edges = canny(gray_norm, sigma=2)

    # ===============================
    # 3-BOSQICH: SCELET (SKELETON)
    # ===============================
    roberts_skeleton = skeletonize(roberts_bin)
    canny_skeleton = skeletonize(canny_edges)

    # ===============================
    # QORA KONTUR – OQ FON
    # ===============================
    def black_contour_white_bg(binary_img):
        return np.where(binary_img, 0, 255)

    roberts_contour = black_contour_white_bg(roberts_bin)
    canny_contour = black_contour_white_bg(canny_edges)

    roberts_skel = black_contour_white_bg(roberts_skeleton)
    canny_skel = black_contour_white_bg(canny_skeleton)

    # ===============================
    # NATIJANI KO‘RSATISH
    # ===============================
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(img_path, fontsize=14)

    # Roberts
    ax[0, 0].imshow(gray, cmap='gray')
    ax[0, 0].set_title("1. Original")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(roberts_contour, cmap='gray')
    ax[0, 1].set_title("2. Roberts Contour")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(roberts_skel, cmap='gray')
    ax[0, 2].set_title("3. Roberts Skeleton")
    ax[0, 2].axis("off")

    # Canny
    ax[1, 0].imshow(gray, cmap='gray')
    ax[1, 0].set_title("1. Original")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(canny_contour, cmap='gray')
    ax[1, 1].set_title("2. Canny Contour")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(canny_skel, cmap='gray')
    ax[1, 2].set_title("3. Canny Skeleton")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
