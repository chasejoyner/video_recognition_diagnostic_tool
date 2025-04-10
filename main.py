import os
import time
import shutil
import threading
import numpy as np

import cv2
import mediapipe as mp
import tkinter as tk


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
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
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
        self.root.title("Pose Recorder")
        self.btn_record = tk.Button(self.root, text="Record 5 Seconds", command=self.record_video, font=("Arial", 14), width=20)
        self.btn_record.pack(padx=20, pady=20)
        self.btn_good = tk.Button(self.root, text="Good", command=self.save_good, font=("Arial", 14), width=20)
        self.btn_bad = tk.Button(self.root, text="Bad", command=self.save_bad, font=("Arial", 14), width=20)

        # Start threading and program
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.root.mainloop()

    def record_video(self):
        """
        Function call to start recording a predetermined long video and save
        """
        if self.recording:
            return
        self.recording = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.last_filename = f"pose_capture_{timestamp}.avi"
        self.out = cv2.VideoWriter(self.last_filename, self.fourcc, self.fps, (self.frame_width, self.frame_height))
        self.start_time = time.time()
        print("Started recording...")

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

    def video_loop(self):
        """
        The main function call for turning on webcam for recording.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

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

            # If recording and video out is detected, finish the recording after elapsed time
            if self.recording and self.out:
                self.out.write(frame)
                elapsed = time.time() - self.start_time
                if elapsed >= self.recordingLength:
                    self.recording = False
                    self.out.release()
                    self.out = None
                    print("Finished recording.")
                    tk.messagebox.showinfo("Done", "Recording complete!")

                    # Show Good/Bad buttons
                    self.btn_good.pack(padx=20, pady=5)
                    self.btn_bad.pack(padx=20, pady=5)

            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

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
