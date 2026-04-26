# Main script to process all images in the assets folder and apply edge detection pipelines
import os
import glob
import cv2
import canny
import complex_canny
import ridge
import accumulator
import complex_accumulator

def process_assets():
    input_folder = './assets'
    
    pipelines = {
        './cannyResults': canny.apply,
        './complexCannyResults': complex_canny.apply,
        './ridgeResults': ridge.apply,
        './accumulatorResults': accumulator.apply,
        './complexAccumulatorResults': complex_accumulator.apply
    }

    for folder in pipelines.keys():
        if not os.path.exists(folder):
            os.makedirs(folder)

    image_paths = glob.glob(os.path.join(input_folder, '*.jpg')) + \
                  glob.glob(os.path.join(input_folder, '*.png'))

    if not image_paths:
        return

    for file_path in image_paths:
        file_name = os.path.basename(file_path)
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        for output_folder, method in pipelines.items():
            try:
                result = method(img) 
                
                save_path = os.path.join(output_folder, file_name)
                cv2.imwrite(save_path, result)
            except Exception:
                continue

if __name__ == "__main__":
    process_assets()