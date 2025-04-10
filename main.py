import os
import time
import shutil
import logging
import numpy as np

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Initalize logger
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class PoseRecorderApp:
    """
    This class uses a connected webcam to record motion and show important bodily nodes, specified by POSE_CONNECTIONS.
    For each captured recording using a tkinter GUI, the user can select good or bad to categorize each recording.
    TODO
    - Implement user accounts via CSV files
    - Implement post recordings analyses, such as average node paths with bands

    Parameters
    ----------
    recordingLength: integer for how long each recording should be
    nodeUpdateThreshold: integer specifying the euclidean distance a node should move before updating frame
    fps: integer specifying the frames per second for recording
    """
    POSE_CONNECTIONS = [(8, 7), (10, 9),
                        (16, 14), (14, 12),
                        (15, 13), (13, 11),
                        (12, 11), (11, 23), (23, 24), (24, 12),
                        (24, 26), (26, 28),
                        (23, 25), (25, 27)]

    def __init__(self, recordingLength=5, nodeUpdateThreshold=10, fps=20):
        # Inputs
        self.recordingLength = recordingLength
        self.nodeUpdateThreshold = nodeUpdateThreshold
        self.fps = fps

        # Initialize pose and recording objects
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_landmarks = {}
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.used_landmark_indices = set()
        for connection in self.POSE_CONNECTIONS:
            self.used_landmark_indices.update(connection)

        # Nuance parameters
        self.recording = False
        self.out = None
        self.start_time = 0
        self.last_filename = None

        # GUI
        self.root = tk.Tk()
        self.root.title("Swing Recorder")
        self.root.configure(bg="#1e1e1e")
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')  # 'clam', 'alt', or 'default'

        # Configure grid layout: 90% for video, 10% for buttons
        self.root.grid_rowconfigure(0, weight=8)  # 90% of window for video
        self.root.grid_rowconfigure(1, weight=2)  # 10% of window for buttons
        self.root.grid_columnconfigure(0, weight=1)  # Full width for both video and buttons

        # Video label (will occupy 90% of the window)
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Fill available space

        self.countdown_label = tk.Label(self.root, text="", font=("Arial", 16), bg="#1e1e1e", fg="white")
        self.countdown_label.grid(row=1, column=0, pady=10)

        # Buttons (placed in the 10% bottom area)
        self.btn_record = tk.Button(self.root, text="Record 5 Seconds", command=self.record_video, font=("Arial", 14),
                                    width=20)
        self.btn_record.grid(row=1, column=0, pady=20)

        self.btn_good = tk.Button(self.root, text="Good", command=self.save_good, font=("Arial", 14), width=20)
        self.btn_bad = tk.Button(self.root, text="Bad", command=self.save_bad, font=("Arial", 14), width=20)

        # Start threading and program

        self.video_loop()
        self.root.mainloop()

    def record_video(self):
        """
        Function call to start recording a predetermined long video and save
        """
        if self.recording:
            return
        self.recording = True
        self.start_time = time.time()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.last_filename = f"pose_capture_{timestamp}.avi"

        self.out = cv2.VideoWriter(self.last_filename, self.fourcc, self.fps, (self.frame_width, self.frame_height))

        logger.info("Started recording...")

        # Update countdown

        self.update_countdown()

    def update_countdown(self):
        """
        Function to update the countdown in the UI during recording
        """
        if self.recording:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.recordingLength - int(elapsed_time))
            self.countdown_label.config(text=f"Recording: {remaining_time} sec")

            if remaining_time > 0:

                self.root.after(1000, self.update_countdown)  # Update every second
            else:

                self.recording = False

                self.out.release()
                self.out = None
                logger.info("Recording completed.")
                self.countdown_label.config(text="Recording done")  # Change text when done
                # Hide the "Record 5 Seconds" button and show the "Good" and "Bad" buttons
                self.btn_record.grid_forget()

                self.btn_good.place(relx=0.4, rely=1, anchor="center")
                self.btn_bad.place(relx=0.6, rely=1, anchor="center")

    def video_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.video_loop)
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # Landmarks detected, draw nodes and connections
        if results.pose_landmarks:

            smoothed_points = {}

            for idx in self.used_landmark_indices:
                landmark = results.pose_landmarks.landmark[idx]
                h, w, _ = frame.shape
                raw_x, raw_y = landmark.x * w, landmark.y * h
                cx, cy = self.smooth_landmark(idx, (raw_x, raw_y))
                smoothed_points[idx] = (cx, cy)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx in smoothed_points and end_idx in smoothed_points:
                    pt1 = smoothed_points[start_idx]
                    pt2 = smoothed_points[end_idx]
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Record video
        if self.recording and self.out:
            self.out.write(frame)
            elapsed = time.time() - self.start_time

            if elapsed >= self.recordingLength:
                self.recording = False

                self.out.release()
                self.out = None
                self.countdown_label.config(text="Recording done")
                # Hide the "Record 5 Seconds" button and show the "Good" and "Bad" buttons
                self.btn_record.grid_forget()
                self.btn_good.place(relx=0.4, rely=1, anchor="center")  # Adjust `relx` to center horizontally
                self.btn_bad.place(relx=0.6, rely=1, anchor="center")

        # Convert to ImageTk format and show in UI

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.video_loop)

    def distance(self, p1, p2):
        """
        Compute Euclidean distance between points p1 and p2
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def smooth_landmark(self, idx, new_point):
        """
        Attempts to smooth node updates to be less jittery
        TODO
        - Determine if we want to do this or only use a nodeUpdateThreshold instead
        """
        new_point = np.array(new_point, dtype=np.float32)
        if idx not in self.prev_landmarks:
            self.prev_landmarks[idx] = new_point
        else:

            if self.distance(self.prev_landmarks[idx], new_point) > self.nodeUpdateThreshold:
                self.prev_landmarks[idx] = new_point

        return tuple(self.prev_landmarks[idx].astype(int))

    def save_to_folder(self, label):

        if not self.last_filename or not os.path.exists(self.last_filename):
            tk.messagebox.showerror("Error", "No video to label.")

            return

        os.makedirs(label, exist_ok=True)
        dest = os.path.join(label, os.path.basename(self.last_filename))
        shutil.move(self.last_filename, dest)
        print(f"Saved video to: {dest}")

        # Reset buttons
        self.btn_good.pack_forget()
        self.btn_bad.pack_forget()
        self.last_filename = None

    def save_good(self):
        self.save_to_folder("good")

    def save_bad(self):
        self.save_to_folder("bad")


if __name__ == "__main__":
    PoseRecorderApp()
