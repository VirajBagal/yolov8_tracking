import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2



def main():
    # line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    # line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    predicted_frames = []
    model = YOLO("yolov8s.pt")
    for result in model.track(source= "/disk5/viraj/yolov8_project/single_lane_more_traffic.mp4", stream=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        traffic_label = "Traffic" if len(labels) > 10 else None
        color = (0, 0, 255)

        if traffic_label:
            cv2.putText(frame, traffic_label, (5, 30), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 1, color = color, thickness = 2)

        # line_counter.trigger(detections=detections)
        # line_annotator.annotate(frame=frame, line_counter=line_counter)
        # line_annotator.annotate(frame=frame)
        predicted_frames.append(frame)
        h, w, c = frame.shape
        size = (w, h)
        if len(predicted_frames) == 50:
            break
        # cv2.imwrite()
        # cv2.imshow("yolov8", frame)

        # if (cv2.waitKey(30) == 27):
        #     break

    out = cv2.VideoWriter('../output/single_lane_more_traffic.mov',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(predicted_frames)):
        out.write(predicted_frames[i])
    out.release()

if __name__ == "__main__":
    main()