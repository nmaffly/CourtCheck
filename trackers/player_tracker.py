from ultralytics import YOLO
import cv2
import pandas as pd

class PlayerTracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        
        player_dict = {}
        for box in results.boxes:
            result = box.xyxy[0].tolist()
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'Player-1' or object_cls_name == 'Player-2':
                player_dict[object_cls_name] = result

        return player_dict
    
    def detect_frames(self, frames):
        object_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            object_detections.append(player_dict)
        return object_detections
    
    def draw_bboxes(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            player_dict = dict(player_dict)  # Convert tuple to dictionary
            for object_name, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"{object_name}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_frames.append(frame)

        print("Frames processed")
        
        return output_frames

    def interpolate_ball(self, object_detections):
        ball_positions = []
        
        for entry in object_detections:
            if "Tennis-ball" in entry:
                ball_positions.append(entry["Tennis-ball"])
            else:
                ball_positions.append([None, None, None, None])
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate().bfill()

        interpolated_positions = df_ball_positions.to_numpy().tolist()

        # Integrate interpolated ball positions back into object_detections
        for i, entry in enumerate(object_detections):
            if "Tennis-ball" not in entry:
                entry["Tennis-ball"] = interpolated_positions[i]
            else:
                entry["Tennis-ball"] = interpolated_positions[i]

        return object_detections



