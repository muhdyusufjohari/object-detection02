import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Set up the Streamlit app
st.title("Object Detection App")
st.write("This app performs real-time object detection using the YOLOv8n model.")

# Set up WebRTC
STUN_SERVER = "stun:stun.l.google.com:19302"
client_settings = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": [STUN_SERVER]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Object detection function
def object_detection(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

# WebRTC video stream callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = object_detection(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC stream
webrtc_streamer(
    key="object_detection",
    mode=WebRtcMode.SENDRECV,
    client_settings=client_settings,
    video_frame_callback=video_frame_callback,
)
