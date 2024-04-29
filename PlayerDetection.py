# installing ultralytics and cv2
from ultralytics import YOLO
import cv2

# load the YOLO model
model = YOLO('yolov8n.pt')

# get image
image_path = './0206.jpg'
image = cv2.imread(image_path)

# detect objects in the image
results = model(image)

# making sure that the result is a list that contatins at least one item
if isinstance(results, list) and len(results) > 0:
    # get the first item from the list and plot result
    results = results[0]
    annotated_image = results.plot()

    # Display the image with detected objects
    cv2.imshow('Detected Objects', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects detected in the image.")
