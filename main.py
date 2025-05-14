import os
import sys
import time
import yaml
import shutil
import logging
import numpy as np
import pandas as pd
import tkinter as tk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
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
    interpolate: boolean to interpolate missing (None) values in self.currentPoseData for each node
    """

    def __init__(self, 
                 recordingLength=5, 
                 nodeUpdateThreshold=10,
                 interpolate=False):

        super().__init__()

        # Get working directory
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")

        # Inputs
        self.recordingLength = recordingLength
        self.nodeUpdateThreshold = nodeUpdateThreshold
        self.interpolate = interpolate

        # Set window icon of GUI
        icon_img = tk.PhotoImage(file=os.path.join(base_path, 'icon.png'))
        self.gui.iconphoto(False, icon_img)

        # Extract yaml contents
        with open(os.path.join(base_path, 'settings.yaml'), 'r') as file:
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

        # Add traces
        self._selectedUserPlotTraceId = ''
        self.selectedUser.trace_add('write', self.on_user_change)
        self.selectedNode.trace_add('write', self.on_node_change)
        self.selectedUser.trace_add('write', lambda *args: None) # Add trace only during analysis frames

        # Initialize pose and recording objects
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.prevLandmarks = {}
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
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


    def videoLoop(self):
        """
        Main loop to capture video and process landmarks
        """
        ret, frame = self.cap.read()
        if not ret:
            # No captured video, show message in UI and wait
            self.textFrameText.config(text='Please connect a webcam and press any key to continue...', fg='red')
            self.videoSection.config(text='Webcam not detected', fg='red')
            self.buttonFrame.pack_forget()
            self.gui.bind('<Key>', self.check_webcam)
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
                self.btnGood.pack(side='left', expand=True, padx=10)
                self.btnBad.pack(side='left', expand=True, padx=10)
                # Enable only Good/Bad buttons for labeling
                self.set_buttons_state('disabled', exclude=[self.btnGood, self.btnBad])
                self.btnGood.config(state='normal')
                self.btnBad.config(state='normal')
                
                # Interpolate pose data to fill in dropped frames
                if self.interpolate:
                    self.interpolate_pose_data()
                
                # Save pose data to user's data
                current_user = self.selectedUser.get()
                if current_user not in self.userData:
                    self.userData[current_user] = []
                df = pd.DataFrame.from_dict(self.currentPoseData)
                df.columns = [self.landmarkDictionary.get(col, 'Unknown node') for col in df.columns]
                df['timestamp'] = datetime.now()
                self.userData[current_user].append(df)
                
                # Reset current pose data for next recording
                self.currentPoseData = {n: [] for n in self.nodes}

        # Only update video display if we're in home view
        if self.videoFrame.winfo_ismapped():
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

        # Continue video loop
        self.gui.after(10, self.videoLoop)


    def recordVideo(self):
        """
        Function call to start recording a predetermined long video and save
        """
        logger.info('Recording video...')
        if self.recording:
            return
        self.set_buttons_state('disabled')
        self.textFrameText.config(text='Recording:')
        self.recording = True
        self.startTime = time.time()
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.lastFilename = f'pose_capture_{timestamp}.avi'
        self.out = cv2.VideoWriter(self.lastFilename, self.fourcc, self.fps, (self.frameWidth, self.frameHeight))

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
        logger.info(f'Saved video to: {dest}')

        # Reset UI to home
        self.btnGood.pack_forget()
        self.btnBad.pack_forget()
        self.btnRecord.pack(side='left', expand=True, padx=10)
        self.textFrameText.config(text='Select an option:')
        self.lastFilename = None
        self.set_buttons_state('normal')


    def saveGood(self):
        label = 'good'
        self.saveToFolder(label)
        current_user = self.selectedUser.get()
        if current_user in self.userData and self.userData[current_user]:
            self.userData[current_user][-1] = (self.userData[current_user][-1], label)


    def saveBad(self):
        label = 'bad'
        self.saveToFolder(label)
        current_user = self.selectedUser.get()
        if current_user in self.userData and self.userData[current_user]:
            self.userData[current_user][-1] = (self.userData[current_user][-1], label)


    def interpolate_pose_data(self):
        """
        Interpolate missing (None) values in self.currentPoseData for each node
        """
        logger.info('Interpolating pose data')
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
        logger.info('Called on node change')
        current_node = self.selectedNode.get()
        logger.info(f'Node changed to {current_node}')
        if self.analysisFrame.winfo_ismapped():
            self.plot_analysis()


    def on_user_change(self, *args):
        """
        Update the plot when the user is changed
        """
        logger.info('Called on user change')
        current_user = self.selectedUser.get()
        logger.info(f'User changed to {current_user}')
        # Only update plot in analysis frame, not in home frame
        if self.analysisFrame.winfo_ismapped():
            self.plot_analysis()
        # Enable record button once user exists
        if current_user and self.videoFrame.winfo_ismapped() and self.btnRecord.cget('state') == 'disabled':
            self.btnRecord.config(state='normal')


    def show_analysis_frame(self):
        """
        Show the analysis frame of the UI
        """
        logger.info('Showing analysis frame')
        self.videoFrame.pack_forget()
        self.textFrame.pack_forget()
        self.buttonFrame.pack_forget()
        self.bottomFrame.pack_forget()

        # Build analysis frame
        self.analysisFrame.pack(fill='both', expand=True, padx=10, pady=10)
        self.nodeDropdown.pack(padx=10, pady=10)
        self.plotSection.pack(fill='both', expand=True, padx=10, pady=10)

        if not self._checkTraceExists(self.selectedUser, self._selectedUserPlotTraceId):
            self._selectedUserPlotTraceId = self.selectedUser.trace_add('write', self.plot_analysis)

        # Add node names to analysis node dropdown
        node_names = [self.landmarkDictionary[n] for n in self.nodes if n in self.landmarkDictionary]
        seen = set()
        filtered_node_names = [x for x in node_names if not (x in seen or seen.add(x))]
        if filtered_node_names:
            self.selectedNode.set(filtered_node_names[0])
            menu = self.nodeDropdown['menu']
            menu.delete(0, 'end')
            for name in filtered_node_names:
                menu.add_command(label=name, command=lambda value=name: self.selectedNode.set(value))

        self.plot_analysis()


    def show_home_frame(self):
        """
        Show the home frame of the UI
        """
        logger.info('Showing home frame')
        self.analysisFrame.pack_forget()
        self.plotSection.pack_forget()

        # Remove plot analysis trace for user selection during home frame
        if self._checkTraceExists(self.selectedUser, self._selectedUserPlotTraceId):
            self.selectedUser.trace_remove('write', self._selectedUserPlotTraceId)

        # Show all home mode frames
        self.bottomFrame.pack(side='bottom', fill='x', padx=10, pady=10)
        self.videoFrame.pack(side='top', fill='both', expand=True)
        self.videoFrame.pack_propagate(False)
        self.textFrame.pack(fill='x', pady=0)
        self.buttonFrame.pack(side='bottom', fill='x', pady=10)


    def plot_analysis(self, *args):
        """
        Main function to plot trajectories for a user and node when analyze button is clicked
        """
        logger.info('Plotting analysis')
        # Clear any existing plot
        if hasattr(self, 'plot_canvas') and self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

        current_user = self.selectedUser.get()
        node_name = self.selectedNode.get()
        if current_user in self.userData and len(self.userData[current_user]) > 0:
            self.plot_trajectories(current_user, nodeName=node_name, parent_frame=self.analysisFrame)
        else:
            logger.info(f'No data available for {current_user}')
            # Create a blank plot with a message if no data exists
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            ax.text(0.5, 0.5, f'No data available for {current_user}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.plot_canvas = FigureCanvasTkAgg(fig, master=self.analysisFrame)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)
            plt.close(fig)


    def plot_trajectories(self, username, nodeName, parent_frame=None):
        """
        Plot all trajectories and their average for a user
        """
        logger.info(f'Plotting trajectories for user {username} and node {nodeName}')
        if parent_frame is None:
            parent_frame = self.plotSection
            
        # Clear previous plot if it exists
        if hasattr(self, 'plot_canvas') and self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        # Separate good and bad trajectories
        good_trajs = []
        bad_trajs = []
        for entry in self.userData[username]:
            if isinstance(entry, tuple):
                df, label = entry
            else:
                df, label = entry, 'good'
            trajectory = np.array(df[nodeName].tolist())
            first_frame = [t for t in trajectory if t[0] is not None and t[1] is not None]
            first_frame = first_frame[0] if first_frame else [None, None]
            centered_trajectory = np.array([
                [point[0] - first_frame[0] if point[0] is not None else None, point[1]]
                for point in trajectory
            ], dtype=np.float32)
            if label == 'good':
                good_trajs.append(centered_trajectory)
            else:
                bad_trajs.append(centered_trajectory)
        # Plot good trajectories
        if good_trajs:
            for traj in good_trajs:
                num_points = len(traj)
                colors = plt.cm.Blues(np.linspace(0, 1, num_points))
                ax.scatter(traj[:, 0], traj[:, 1], color=colors, zorder=1)
        # Plot bad trajectories
        if bad_trajs:
            for traj in bad_trajs:
                num_points = len(traj)
                colors = plt.cm.Purples(np.linspace(0, 1, num_points))
                ax.scatter(traj[:, 0], traj[:, 1], color=colors, zorder=1)
        # Calculate and plot average trajectory (for good only)
        if good_trajs:
            good_trajs_arr = np.array(good_trajs, dtype=np.float32)
            avg_trajectory = []
            for i in range(good_trajs_arr.shape[1]):
                if np.all(np.isnan(good_trajs_arr[:, i, :])):
                    avg_trajectory.append([None, None])
                else:
                    avg_trajectory.append(np.nanmean(good_trajs_arr[:, i, :], axis=0))
            avg_trajectory = np.array(avg_trajectory)
            ax.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='red', linewidth=2, label='Average Good', zorder=2)
        # Calculate and plot average trajectory (for bad only)
        if bad_trajs:
            bad_trajs_arr = np.array(bad_trajs, dtype=np.float32)
            avg_trajectory = []
            for i in range(bad_trajs_arr.shape[1]):
                if np.all(np.isnan(bad_trajs_arr[:, i, :])):
                    avg_trajectory.append([None, None])
                else:
                    avg_trajectory.append(np.nanmean(bad_trajs_arr[:, i, :], axis=0))
            avg_trajectory = np.array(avg_trajectory)
            ax.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], color='purple', linewidth=2, label='Average Bad', zorder=2)
        ax.set_title(f'Trajectories for {nodeName} for user {username}')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((-1, 1))
        ax.set_ylim((0, 1))
        ax.invert_yaxis()

        # Add custom handles to the legend
        good_grad = tuple([Line2D([0, 0], [0, 0], color=plt.cm.Blues(i), linewidth=2) for i in np.linspace(0, 1, 10)])
        bad_grad = tuple([Line2D([0, 0], [0, 0], color=plt.cm.Purples(i), linewidth=2) for i in np.linspace(0, 1, 10)])
        custom_handles = [good_grad, bad_grad]
        custom_labels = ['Good (Start → Finish)', 'Bad (Start → Finish)']
        ax.legend(custom_handles + ax.get_legend_handles_labels()[0],
                  custom_labels + ax.get_legend_handles_labels()[1],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
        ax.grid(True)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

    
    def _checkTraceExists(self, object, trace_id):
        """
        Check if a trace exists on an object
        """
        for _, cbname in object.trace_info():
            if cbname == trace_id:
                return True
        return False


    def cleanup_and_exit(self):
        """
        Clean up resources and exit the program gracefully
        """
        logger.info('Cleaning up resources and exiting...')
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        self.gui.quit()
        self.gui.destroy()


    def run(self):
        self.videoLoop()
        self.gui.mainloop()


    def check_webcam(self, event):
        """
        Check if webcam is now available and restart video loop if it is
        """
        self.gui.unbind('<Key>')
        # Release the current capture if it exists
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        # Try to initialize new capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        ret, _ = self.cap.read()
        if ret:
            self.textFrameText.config(text='Select an option:', fg='white')
            self.videoSection.config(text='', fg='white')
            self.buttonFrame.pack(side='bottom', fill='x', pady=10)
            self.videoLoop()
        else:
            self.gui.after(1000, lambda: self.check_webcam(None))


    def set_buttons_state(self, state, exclude=None):
        """
        Set the state of all main buttons
        Optionally exclude some buttons from being changed
        """
        exclude = exclude or []
        buttons = [self.btnRecord, self.btnGood, self.btnBad]
        for btn in buttons:
            if btn not in exclude:
                btn.config(state=state)


if __name__ == '__main__':
    app = PoseRecorderApp()
    app.run()
