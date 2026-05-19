from tqdm import tqdm
import numpy as np
import glob
import cv2

def crop_cell_from_frames(image_path: str,
                          output_dir: str = None,
                          chunk: int = None,
                          overwrite_file: bool = False,
                          verbose: bool = False):

    # Checks for output_dir or overwrite_file
    if not output_dir and not overwrite_file:
        print("[ERROR]: OUTPUT DIRECTORY MUST BE PROVIDED OR ORIGINAL FILE OVERWRITTEN")
        return None

    # Reads image file
    img = cv2.imread(image_path)

    # Converts from RGB/BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper blue HSV --> Hue, Saturation, Value
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Applies binary, color-based mask on the HSV image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Removes further image noise using median blur, 5x5 grid
    mask = cv2.medianBlur(mask, 5)

    # Finds contours: EXTERNAL and COMPRESSED
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[ERROR]: NO BLUE CONTOUR FOUND\nFRAME: {image_path}")
        return None

    # Finds largest contour
    contour = max(contours, key=cv2.contourArea)

    # Gets contour bounding box
    x, y, w, h = cv2.boundingRect(contour)

    if chunk:
        xy = chunk // 2

        # Increases the contour
        x, y, w, h = x-xy, y-xy, w+chunk, h+chunk

    # Crops original image
    crop = img[y:y+h, x:x+w]

    # Saves the cropped file
    if overwrite_file: # Overwrites original image file

        # Saves crop
        cv2.imwrite(image_path, crop)
        if verbose:
            print(f"{image_path} Overwritten Successfully!")
    else: # Creates new file

        # Gets crop file name
        crop_output_path = output_dir + "/cropped_" + (image_path.split('/')[-1])

        # Saves crop
        cv2.imwrite(crop_output_path, crop)
        if verbose:
            print(f"CROP SAVED AT {crop_output_path}")

def crop_all_images(extracted_frames_dir: str,
                    output_dir: str = None,
                    chunk: int = None,
                    overwrite_file: bool = False,
                    verbose: bool = False):

    # Sets the entire path
    full_path = extracted_frames_dir + "/*"

    # Lists all image files
    files = glob.glob(full_path)

    # Checks for files
    if not files:
        print("[ERROR]: EXTRACTED FRAMES DIRECTORY IS EMPTY!")
        return None

    for file_path in tqdm(files):
        crop_cell_from_frames(image_path=file_path, output_dir=output_dir,
                              chunk=chunk, overwrite_file=overwrite_file, verbose=verbose)


if __name__ == "__main__":
    ORIGINAL_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset/video_frames_5m"
    CROPPED_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset/cropped_frames_5m"

    crop_all_images(extracted_frames_dir=ORIGINAL_FRAMES_DIR,
                    output_dir=CROPPED_FRAMES_DIR,
                    verbose=False,
                    chunk=10)
