import cv2, os

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

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames to save.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get the frame dimensions
    height, width, _ = output_video_frames[0].shape

    # Try different codecs if necessary
    codecs = ['MJPG', 'XVID', 'mp4v', 'DIVX']
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if out.isOpened():
            break
        else:
            out.release()

    if not out.isOpened():
        raise ValueError(f"Failed to open VideoWriter for file: {output_video_path} with any codec")

    for i, frame in enumerate(output_video_frames):
        if frame is None or frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError(f"Frame {i} is invalid or has incorrect dimensions.")
        out.write(frame)

    out.release()

    print(f"Video saved successfully to {output_video_path}")


