import os
import time
import yaml
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from gui import PoseGUIApp

# Initalize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseRecorderApp(PoseGUIApp):
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
    interpolate: boolean to interpolate missing (None) values in self.currentPoseData for each node
    """

    def __init__(self, 
                 recordingLength=5, 
                 nodeUpdateThreshold=10, 
                 fps=20,
                 interpolate=False):

        super().__init__()

        # Inputs
        self.recordingLength = recordingLength
        self.nodeUpdateThreshold = nodeUpdateThreshold
        self.fps = fps

        # Extract yaml contents
        with open('settings.yaml', 'r') as file:
            data = yaml.safe_load(file)
        self.landmarkDictionary = data['LANDMARKS_DICTIONARY']
        self.poseConnections = [tuple(pc) for pc in data['POSE_CONNECTIONS']]
        self.nodes = [n for pc in self.poseConnections for n in pc]

        # Initialize pose and recording objects
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.prevLandmarks = {}
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.usedLandmarkIndices = set(c for pc in self.poseConnections for c in pc)

        # Add button commands
        self.btnRecord.config(command=self.recordVideo)
        self.btnGood.config(command=self.saveGood)
        self.btnBad.config(command=self.saveBad)

        # Run variables
        self.recording = False
        self.out = None
        self.startTime = 0
        self.lastFilename = None
        self.currentPoseData = {n: [] for n in self.nodes}
        self.userData = {}  # Will store pose data by username: {username: {node_id: [(x1,y1), (x2,y2), ...]}}


    def recordVideo(self):
        """
        Function call to start recording a predetermined long video and save
        """
        if self.recording:
            return
        self.textFrameText.config(text='Recording:')
        self.recording = True
        self.startTime = time.time()
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.lastFilename = f'pose_capture_{timestamp}.avi'
        self.out = cv2.VideoWriter(self.lastFilename, self.fourcc, self.fps, (self.frameWidth, self.frameHeight))
        logger.info('Started recording...')
        self.btnRecord.pack_forget()
        self.btnGood.pack_forget()
        self.btnBad.pack_forget()
        self.countdownText.pack(side='top', expand=True, padx=10)
        self.num_frames = int(self.fps * (self.recordingLength - 1))
        self.frame_counter = 0
        self.currentPoseData = {n: [None] * self.num_frames for n in self.nodes}
        self.updateCountdown()


    def updateCountdown(self):
        """
        Function to update the countdown in the UI during recording
        """
        if self.recording:
            elapsedTime = time.time() - self.startTime
            remainingTime = max(0, self.recordingLength - int(elapsedTime))
            self.countdownText.config(text=f'{remainingTime} sec')

            if remainingTime > 0:
                self.gui.after(1000, self.updateCountdown)


    def videoLoop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.gui.after(10, self.videoLoop)
            return

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        # Landmarks detected, draw nodes and connections
        if results.pose_landmarks:
            smoothedPoints = {}
            for idx in self.usedLandmarkIndices:
                landmark = results.pose_landmarks.landmark[idx]
                h, w, _ = frame.shape
                rawX, rawY = landmark.x * w, landmark.y * h
                cx, cy = self.smoothLandmark(idx, (rawX, rawY))
                smoothedPoints[idx] = (cx, cy)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                if self.recording and self.frame_counter < self.num_frames:
                    self.currentPoseData[idx][self.frame_counter] = (landmark.x, landmark.y)

            for connection in self.poseConnections:
                startIdx, endIdx = connection
                if startIdx in smoothedPoints and endIdx in smoothedPoints:
                    pt1 = smoothedPoints[startIdx]
                    pt2 = smoothedPoints[endIdx]
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Record video
        if self.recording and self.out:
            self.out.write(frame)
            if self.frame_counter < self.num_frames:
                self.frame_counter += 1
            elapsed = time.time() - self.startTime
            if elapsed >= self.recordingLength or self.frame_counter >= self.num_frames:
                self.recording = False
                self.out.release()
                self.out = None
                self.btnRecord.pack_forget()
                self.countdownText.pack_forget()
                self.textFrameText.config(text='Label the hit:')
                self.textFrameText.pack()
                self.btnGood.pack(side='left', expand=True, padx=10)
                self.btnBad.pack(side='left', expand=True, padx=10)
                
                # Interpolate pose data to fill in dropped frames
                if self.interpolate:
                    self.interpolate_pose_data()
                
                current_user = self.selectedOption.get()
                if current_user not in self.userData:
                    self.userData[current_user] = []
                df = pd.DataFrame.from_dict(self.currentPoseData)
                df.columns = [self.landmarkDictionary.get(col, 'Unknown node') for col in df.columns]
                df['timestamp'] = datetime.now()
                self.userData[current_user].append(df)
                # Reset current pose data for next recording
                self.currentPoseData = {n: [] for n in self.nodes}
                if len(self.userData[current_user]) == 3:
                    self.plot_wrist_trajectories(current_user)
            
        # Dynamically resize video to fit the video section frame
        labelWidth = self.videoSection.winfo_width()
        labelHeight = self.videoSection.winfo_height()
        if labelWidth > 0 and labelHeight > 0:
            frame = cv2.resize(frame, (labelWidth, labelHeight), interpolation=cv2.INTER_AREA)

        # Convert to ImageTk format and show in UI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.videoSection.imgtk = imgtk
        self.videoSection.configure(image=imgtk)
        self.gui.after(10, self.videoLoop)


    def distance(self, p1, p2):
        """
        Compute the Euclidean distance between points p1 and p2
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))


    def smoothLandmark(self, idx, newPoint):
        """
        Attempts to smooth node updates to be less jittery
        TODO
        - Determine if we want to do this or only use a nodeUpdateThreshold instead
        """
        newPoint = np.array(newPoint, dtype=np.float32)
        if idx not in self.prevLandmarks:
            self.prevLandmarks[idx] = newPoint
        else:
            if self.distance(self.prevLandmarks[idx], newPoint) > self.nodeUpdateThreshold:
                self.prevLandmarks[idx] = newPoint

        return tuple(self.prevLandmarks[idx].astype(int))


    def saveToFolder(self, label):
        os.makedirs(label, exist_ok=True)
        dest = os.path.join(label, os.path.basename(self.lastFilename))
        shutil.move(self.lastFilename, dest)
        print(f'Saved video to: {dest}')

        # Reset
        self.btnGood.pack_forget()
        self.btnBad.pack_forget()
        self.btnRecord.pack(side='left', expand=True, padx=10)
        self.textFrameText.config(text='Select an option:')
        self.lastFilename = None


    def saveGood(self):
        self.saveToFolder('good')


    def saveBad(self):
        self.saveToFolder('bad')


    def run(self):
        self.videoLoop()
        self.gui.mainloop()


    def plot_wrist_trajectories(self, username):
        """Plot all wrist trajectories and their average for a user"""
        plt.figure(figsize=(12, 8))
        # import pickle
        # with open('trajectories.pkl', 'rb') as f:
        #     self.userData[username] = pickle.load(f)
        
        # Extract x,y coordinates from each recording's Right wrist column
        wrist_trajectories = []
        for df in self.userData[username]:
            # Each point in Right wrist column is a tuple of (x,y)
            trajectory = np.array(df['Right wrist'].tolist())
            wrist_trajectories.append(trajectory)
            # Plot individual trajectory
            plt.scatter(trajectory[1:-1, 0], trajectory[1:-1, 1], color='black', zorder=1)
            plt.scatter(trajectory[0, 0], trajectory[0, 1], color='purple', label='Start', zorder=2)
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', label='End', zorder=2)

        # Calculate and plot average trajectory
        wrist_trajectories = np.array(wrist_trajectories)
        avg_trajectory = np.mean(wrist_trajectories, axis=0)
        plt.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], 'r-', linewidth=2, label='Average Trajectory', zorder=3)

        # Customize plot
        plt.title(f'Right Wrist Trajectories for {username}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots/wrist_trajectory_{username}_{timestamp}.png')
        plt.close()
        
        logger.info(f'Saved wrist trajectory plot for {username}')


    def interpolate_pose_data(self):
        """Interpolate missing (None) values in self.currentPoseData for each node."""
        for node, coords in self.currentPoseData.items():
            coords_arr = np.array([c if c is not None else (np.nan, np.nan) for c in coords], dtype=float)
            # Interpolate x and y separately
            for dim in [0, 1]:
                arr = coords_arr[:, dim]
                nans = np.isnan(arr)
                if np.any(~nans):
                    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
                coords_arr[:, dim] = arr
            self.currentPoseData[node] = [tuple(xy) for xy in coords_arr]


if __name__ == '__main__':
    app = PoseRecorderApp()
    app.run()
