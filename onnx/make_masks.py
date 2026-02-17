import os
import glob
import cv2
import numpy as np

def make_masks(dir_path: str):
    base_image_path = '../data/AITEX_Fabric_Image_Database/Defect_images'
    base_mask_path = '../data/AITEX_Fabric_Image_Database/Mask_images'
    files = glob.glob(f"{dir_path}/*.png")
    for file in files:
        fn = file.split('.')[0].split('/')[-1]
        print(fn)
        image_path = os.path.join(base_image_path, fn + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        mask_path = os.path.join(base_mask_path, fn + '_mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        crop_mask = np.zeros_like(image)
        crop_mask[mask > 0] = image[mask > 0]
        cv2.imwrite(f"{dir_path}/{fn}_gt_mask.png", cv2.cvtColor(crop_mask, cv2.COLOR_RGB2BGR))


        pred_path = file
        pred = cv2.imread(pred_path)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        crop_pred = np.zeros_like(image)

        # boolean masks for each color
        red_mask = (pred[:, :, 0] == 255) & (pred[:, :, 1] == 0) & (pred[:, :, 2] == 0)
        green_mask = (pred[:, :, 0] == 0) & (pred[:, :, 1] == 255) & (pred[:, :, 2] == 0)
        blue_mask = (pred[:, :, 0] == 0) & (pred[:, :, 1] == 0) & (pred[:, :, 2] == 255)

        # copy original pixels
        crop_pred[red_mask] = image[red_mask]
        crop_pred[green_mask] = image[green_mask]
        crop_pred[blue_mask] = image[blue_mask]

        # apply tint (scale channels)
        scale = 0.7
        crop_pred[red_mask] = crop_pred[red_mask] * np.array([1.0, scale, scale])
        crop_pred[green_mask] = crop_pred[green_mask] * np.array([scale, 1.0, scale])
        crop_pred[blue_mask] = crop_pred[blue_mask] * np.array([scale, scale, 1.0])

        crop_pred = crop_pred.astype(np.uint8)
        cv2.imwrite(f"{dir_path}/{fn}_pred_viz.png", cv2.cvtColor(crop_pred, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    make_masks('valid')
    make_masks('test')