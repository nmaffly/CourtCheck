from ultralytics import YOLO
import os
import cv2

def images_to_video(directory, output_video_file, fps=24):
    # Obtain list of image files in the directory sorted by name
    images = [img for img in os.listdir(directory) if img.endswith(".jpg")]
    images.sort()

    # Read the first image to obtain width and height
    frame = cv2.imread(os.path.join(directory, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'x264' might be available
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(directory, image))
        video.write(frame)  # Write out frame to video

    video.release()

model = YOLO('weights/best_yolo.pt')

directory = 'Dataset/game1/Clip5'
output_video_file = 'Dataset/game1/Clip5/G2Clip5_video.mp4'
images_to_video(directory, output_video_file)

result = model.track(output_video_file, conf=0.2, save=True)
print(result)
# for filename in os.listdir(directory):
#      if filename.endswith(".jpg"):
#          result = model.predict(os.path.join(directory, filename), save=True)
#          print(result)

