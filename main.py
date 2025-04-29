import os
import time
import yaml
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
                 fps=30,
                 interpolate=False):

        super().__init__()

        # Inputs
        self.recordingLength = recordingLength
        self.nodeUpdateThreshold = nodeUpdateThreshold
        self.fps = fps
        self.interpolate = interpolate

        # Extract yaml contents
        with open('settings.yaml', 'r') as file:
            data = yaml.safe_load(file)
        self.landmarkDictionary = data['LANDMARKS_DICTIONARY']
        self.poseConnections = [tuple(pc) for pc in data['POSE_CONNECTIONS']]
        self.nodes = [n for pc in self.poseConnections for n in pc]

        # Only allow node names that appear in pose_connections
        node_names = [self.landmarkDictionary[n] for n in self.nodes if n in self.landmarkDictionary]
        seen = set()
        filtered_node_names = [x for x in node_names if not (x in seen or seen.add(x))]
        if filtered_node_names:
            self.selectedNode.set(filtered_node_names[0])
        self.selectedNode.trace_add('write', self.on_node_change)
        self.selectedOption.trace_add('write', self.on_user_change)

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
        self.userData = {}


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
        self.currentPoseData = {n: [(None, None)] * self.num_frames for n in self.nodes}
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
        """
        Main loop to capture video and process landmarks
        """
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
                if len(self.userData[current_user]) >= 0:
                    self.plot_trajectories(current_user, nodeName=self.selectedNode.get())
            
            # Hide the plot after labeling
            if hasattr(self, 'plot_canvas') and self.plot_canvas is not None:
                self.plot_canvas.get_tk_widget().destroy()
                self.plot_canvas = None
            
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


    def plot_trajectories(self, username, nodeName, parent_frame=None):
        """
        Plot all trajectories and their average for a user
        """
        if parent_frame is None:
            parent_frame = self.plotFrame
        # Clear previous plot if it exists
        if hasattr(self, 'plot_canvas') and self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        # Extract x,y coordinates from each recording's node column
        trajectories = []
        for df in self.userData[username]:
            trajectory = np.array(df[nodeName].tolist())
            # Horizontal center the trajectory so the first point is at the origin
            # Probably we will make ball the relative point to center around
            first_frame = [t for t in trajectory if t[0] is not None and t[1] is not None]
            first_frame = first_frame[0] if first_frame else [None, None]
            centered_trajectory = np.array([
                [
                    point[0] - first_frame[0] if point[0] is not None else None,
                    point[1]
                ]
                for point in trajectory
            ], dtype=np.float32)
            trajectories.append(centered_trajectory)
            ax.scatter(centered_trajectory[1:-1, 0], centered_trajectory[1:-1, 1], color='black', zorder=1)
            ax.scatter(centered_trajectory[0, 0], centered_trajectory[0, 1], color='purple', zorder=2)
            ax.scatter(centered_trajectory[-1, 0], centered_trajectory[-1, 1], color='green', zorder=2)

        # Calculate and plot average trajectory
        trajectories = np.array(trajectories, dtype=np.float32)
        avg_trajectory = []
        for i in range(trajectories.shape[1]):
            if np.all(np.isnan(trajectories[:, i, :])):
                avg_trajectory.append([None, None])
            else:
                avg_trajectory.append(np.nanmean(trajectories[:, i, :], axis=0))
        avg_trajectory = np.array(avg_trajectory)
        ax.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], 'r-', linewidth=2, label='Average Trajectory', zorder=4)

        ax.set_title(f'Trajectories for {nodeName} for user {username}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((-1, 1))
        ax.set_ylim((0, 1))
        ax.invert_yaxis()
        custom_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Start'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Finish')
        ]
        ax.legend(handles=custom_handles + ax.get_legend_handles_labels()[0])
        ax.grid(True)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
        logger.info(f'Displayed trajectories plot for user {username} and node {nodeName} in UI')


    def interpolate_pose_data(self):
        """
        Interpolate missing (None) values in self.currentPoseData for each node
        """
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


    def on_node_change(self, *args):
        """
        Update the plot when the node is changed
        """
        current_user = self.selectedOption.get()
        if current_user in self.userData and len(self.userData[current_user]) > 0:
            self.plot_trajectories(current_user, nodeName=self.selectedNode.get(), parent_frame=self.plotFrame)


    def on_user_change(self, *args):
        """
        Update the plot when the user is changed
        """
        current_user = self.selectedOption.get()
        if current_user in self.userData and len(self.userData[current_user]) > 0:
            self.plot_trajectories(current_user, nodeName=self.selectedNode.get(), parent_frame=self.plotFrame)


    def show_analysis_frame(self):
        """
        Show the analysis frame of the UI
        """
        self.videoFrame.pack_forget()
        self.plotFrame.pack_forget()
        self.analyzeButton.config(text='Home', command=self.show_home_frame)

        node_names = [self.landmarkDictionary[n] for n in self.nodes if n in self.landmarkDictionary]
        seen = set()
        filtered_node_names = [x for x in node_names if not (x in seen or seen.add(x))]
        if filtered_node_names:
            self.analysisNodeVar.set(filtered_node_names[0])
            menu = self.analysisNodeDropdown['menu']
            menu.delete(0, 'end')
            for name in filtered_node_names:
                menu.add_command(label=name, command=lambda value=name: self.analysisNodeVar.set(value))
        
        # Add trace to analysis node selection
        self.analysisNodeVar.trace_add('write', self.plot_analysis)

        # Build analysis frame and plot trajectories
        self.analysisFrame.pack(fill='both', expand=True, padx=10, pady=10)
        self.plot_analysis()


    def show_home_frame(self):
        """
        Show the home frame of the UI
        """
        self.analysisFrame.pack_forget()
        self.analyzeButton.config(text='Analyze', command=self.show_analysis_frame)
        self.videoFrame.pack(side='top', fill='both', expand=True)
        self.plotFrame.pack(fill='both', expand=False, padx=10, pady=10)
        

    def plot_analysis(self, *args):
        """
        Main function to plot trajectories for a user and node when analyze button is clicked
        """
        current_user = self.selectedOption.get()
        node_name = self.analysisNodeVar.get()
        if current_user in self.userData and len(self.userData[current_user]) > 0:
            self.plot_trajectories(current_user, nodeName=node_name, parent_frame=self.analysisFrame)


if __name__ == '__main__':
    app = PoseRecorderApp()
    app.run()
