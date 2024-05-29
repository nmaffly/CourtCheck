from utils import (images_to_video, 
                   read_video, 
                   save_video
                   )

from trackers import BallTracker, PlayerTracker
import os

def main():
    game = '2'
    clip = '4'
    directory = f'Dataset/game{game}/Clip{clip}'
    input_video_file = f'Dataset/game{game}/Clip{clip}/G{game}Clip{clip}_video.mp4'
    images_to_video(directory, input_video_file)

    output_video_file = f'output_videos/output_video_G{game}C{clip}.avi'

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    frames = read_video(input_video_file)

    ball_tracker = BallTracker('weights/model_best.pt')

    if not os.path.exists(f'tracker_stubs/G{game}C{clip}_stub.pkl'):
        use_stubs = False
    else:
        use_stubs = True
    
    ball_detections, dists = ball_tracker.infer_tracknet_model(frames, use_stubs=use_stubs, stub_path=f'tracker_stubs/G{game}C{clip}_stub.pkl')
    ball_detections = ball_tracker.remove_outliers(dists, ball_detections)
    ball_detections = ball_tracker.interpolate(ball_detections)
    output_frames = ball_tracker.draw_ball_detections(frames, ball_detections)

    player_tracker = PlayerTracker('weights/best_yolo.pt')

    player_detections = player_tracker.detect_frames(frames)
    output_frames = player_tracker.draw_bboxes(output_frames, player_detections)

    save_video(output_frames, output_video_file)

if __name__ == "__main__":
    main()