import torch, cv2, pickle
import numpy as np
from tqdm import tqdm
from model import *
import pandas as pd
from scipy.spatial import distance


class BallTracker:
    def __init__(self, model_path):
        model = BallTrackerNet()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        self.model = model
 
    def infer_tracknet_model(self, frames, use_stubs=False, stub_path=None):
        height = 360
        width = 640
        dists = [-1]*2
        if use_stubs and stub_path is not None:
           ball_detections = pickle.load(open(stub_path, 'rb'))
           for num in tqdm(range(2, len(frames))):
                if ball_detections[-1][0] and ball_detections[-2][0]:
                    dist = distance.euclidean(ball_detections[-1], ball_detections[-2])
                else:
                    dist = -1
                dists.append(dist) 
        else:
            ball_detections = [(None,None)]*2
            for num in tqdm(range(2, len(frames))):
                # resize three images at a time for model to process
                img = cv2.resize(frames[num], (width, height))
                img_prev = cv2.resize(frames[num-1], (width, height))
                img_preprev = cv2.resize(frames[num-2], (width, height))
                imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
                imgs = imgs.astype(np.float32)/255.0
                imgs = np.rollaxis(imgs, 2, 0)
                inp = np.expand_dims(imgs, axis=0)

                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                x_pred, y_pred = self.postprocess(output)
                ball_detections.append((x_pred, y_pred))

                if ball_detections[-1][0] and ball_detections[-2][0]:
                    dist = distance.euclidean(ball_detections[-1], ball_detections[-2])
                else:
                    dist = -1
                dists.append(dist)  

            if stub_path is not None:
                pickle.dump(ball_detections, open(stub_path, 'wb'))

        return ball_detections, dists

    def postprocess(self, feature_map, scale=2):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                maxRadius=7)
        x,y = None, None
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y

    def remove_outliers(self, dists, ball_detections, max_dist=100):
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i+1] > max_dist) | (dists[i+1] == -1):
                ball_detections[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_detections[i-1] = (None, None)
        
        return ball_detections 
    
    def interpolate(self, ball_detections):
        ball_positions = []
        for entry in ball_detections[:2]:
            ball_positions.append(entry)
    
        for entry in ball_detections[2:]:
            if entry[0] is not None and entry[1] is not None:
                ball_positions.append(entry)
            else:
                ball_positions.append([None, None])
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x', 'y'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        interpolated_positions = ball_positions[:2] + df_ball_positions.to_numpy().tolist()

        for i, entry in enumerate(ball_detections):
            if not entry[0] or not entry[1]:
                ball_detections[i] = (interpolated_positions[i][0], interpolated_positions[i][1])

        return ball_detections

    def draw_ball_detections(self, frames, ball_detections):
        output_frames = []
        for frame, ball_detection in zip(frames, ball_detections):
            x, y = ball_detection
            if x and y:
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)
            output_frames.append(frame)
        return output_frames


