import streamlit as st
from ultralytics import YOLO
import supervision as sv
import cv2
import os
import subprocess

# if number of detected vehicles are greater than the following threshold, then there is said to be Traffic
TRAFFIC_THRESHOLD = 10

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def detect_traffic(video):
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    predicted_frames = []
    model = YOLO("yolov8s.pt")

    save_dir = "uploaded"
    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, video.name)
    save_uploaded_file(video, save_path)

    pred_save_dir = "output"
    os.makedirs(pred_save_dir, exist_ok = True)
    pred_save_path = os.path.join(pred_save_dir, video.name)

    for result in model.track(source= save_path, stream=True):
        
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

        traffic_label = "Traffic" if len(labels) > TRAFFIC_THRESHOLD else None
        color = (0, 0, 255)


        # write number of detected vehicles on the image
        h, w = frame.shape[:2]
        fontscale = (h // 720)+ 1
        text_start_y_coordinate = h * 50 // 720

        if traffic_label:
            cv2.putText(frame, traffic_label, (w // 2, text_start_y_coordinate), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = fontscale, color = color, thickness = 2)

        cv2.putText(frame, f"Num Vehicles: {len(labels)}", (5, text_start_y_coordinate), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = fontscale, color = color, thickness = 2)

        predicted_frames.append(frame)
        h, w, c = frame.shape
        size = (w, h)
        if len(predicted_frames) == 100:
            break

    # first save it as mp4 using mp4v codec. But streamlit does not support it. So after this, need to use ffmpeg  to save as mp4 using h264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pred_save_path, fourcc, 15, size)
    for i in range(len(predicted_frames)):
        out.write(predicted_frames[i])
    out.release()

    ## Reencodes video to H264 using ffmpeg. Streamlit needs mp4 encoded with h264 codec. Above we used mp4v. Opencv doesnt support h264
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i {pred_save_path} -c:v libx264 {pred_save_path.replace('.mp4', '_h264.mp4')}".split(" "))
    return pred_save_path.replace('.mp4', '_h264.mp4')

def main():
    st.title("Vehicle Detection")

    # Display a file uploader to upload the video
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file:
        save_path = detect_traffic(uploaded_file)
        
        if save_path:
            current_file_path = os.getcwd()
            full_path = os.path.join(current_file_path, save_path)
            st.video(full_path)

if __name__ == "__main__":
    main()
        


