import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Initalize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseGUIApp:
    """
    This class creates a TKinter GUI to be used by the pose recorder class
    """

    def __init__(self):
        # Initialize GUI
        self.gui = tk.Tk()
        self.gui.minsize(800, 600)
        self.gui.title('Swing Recorder')
        self.gui.configure(bg='#1e1e1e')
        self.guiStyle = ttk.Style(self.gui)
        self.guiStyle.theme_use('clam')

        # Create top frame for controls
        self.topFrame = tk.Frame(self.gui, bg='#1e1e1e', height=40)
        self.topFrame.pack(side='top', fill='x', padx=10, pady=10)
        self.topFrame.pack_propagate(False)

        # Create middle frame for video and plot
        self.middleFrame = tk.Frame(self.gui, bg='#1e1e1e', height=500)
        self.middleFrame.pack(side='top', fill='both', expand=True, padx=10, pady=10)
        self.middleFrame.pack_propagate(False)

        # Create bottom frame for text and buttons
        self.bottomFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.bottomFrame.pack(side='bottom', fill='x')

        # Add controls to top frame
        self.newUserButton = tk.Button(self.topFrame, text='New User', font=('Arial', 12))
        self.newUserButton.pack(side='left', padx=5)
        self.userNames = []
        self.selectedUser = tk.StringVar()
        self.userDropdownFrame = tk.Frame(self.topFrame, bg='#1e1e1e', width=100, height=40)
        self.userDropdownFrame.pack(side='left', padx=5)
        self.userDropdownFrame.pack_propagate(False)
        self.userDropdown = ttk.OptionMenu(self.userDropdownFrame, self.selectedUser, '', *self.userNames)
        self.userDropdown.pack(expand=True)
        self.analyzeButton = tk.Button(self.topFrame, text='Analyze', font=('Arial', 12))
        self.analyzeButton.pack(side='right', padx=5)       

        # Create video frame in middle frame
        self.videoFrame = tk.Frame(self.middleFrame, bg='#1e1e1e')
        self.videoFrame.pack(side='top', fill='both', expand=True)
        self.videoFrame.pack_propagate(False)
        self.videoSection = tk.Label(self.videoFrame)
        self.videoSection.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Analysis frame which shows up when the analyze button is clicked
        self.analysisFrame = tk.Frame(self.middleFrame, bg='#1e1e1e')
        self.plotSection = tk.Frame(self.analysisFrame, bg='#1e1e1e')
        self.selectedNode = tk.StringVar()
        self.nodeDropdown = ttk.OptionMenu(self.analysisFrame, self.selectedNode, '')

        # Create text frame in bottom frame
        self.textFrame = tk.Frame(self.bottomFrame, bg='#1e1e1e')
        self.textFrame.pack(fill='x', pady=0)
        self.textFrameText = tk.Label(self.textFrame, text='Select an option:', font=('Arial', 14), fg='white', bg='#1e1e1e')
        self.textFrameText.pack()

        # Create button frame in bottom frame
        self.buttonFrame = tk.Frame(self.bottomFrame, height=100, bg='#1e1e1e')
        self.buttonFrame.pack(side='bottom', fill='x', pady=10)
        self.buttonFrame.pack_propagate(False)
        self.btnRecord = tk.Button(self.buttonFrame, text='Record 5 Seconds', font=('Arial', 14), width=20)
        self.btnRecord.pack(side='left', expand=True, padx=10)
        self.btnGood = tk.Button(self.buttonFrame, text='Good', font=('Arial', 14), width=20)
        self.btnBad = tk.Button(self.buttonFrame, text='Bad', font=('Arial', 14), width=20)
        self.countdownText = tk.Label(self.buttonFrame, font=('Arial', 14), fg='white', bg='#1e1e1e')


    def updateUserDropdownOptions(self):
        """
        Updates the user names in the dropdown menu
        """
        menu = self.userDropdown['menu']
        menu.delete(0, 'end')
        for option in self.userNames:
            menu.add_radiobutton(label=option, variable=self.selectedUser, value=option)
        
        if self.userNames:
            self.userDropdown.pack()
            if not self.selectedUser.get():
                self.selectedUser.set(self.userNames[0])
        else:
            self.userDropdown.pack_forget()


    def addNewUser(self):
        """
        Handles adding a new user to the dropdown
        """
        while True:
            new_name = simpledialog.askstring('New User', "Enter the new user's name:", parent=self.gui)

            if new_name is None:
                logger.info('New user entry cancelled.')
                break

            new_name = new_name.strip()
            if not new_name:
                logger.info('Empty name entered.')
                messagebox.showerror('Error', 'No name was entered.')
                continue

            if new_name in self.userNames:
                logger.info(f'User {new_name} already exists.')
                messagebox.showerror('Error', 'User already exists, please enter a different name.')
                continue

            # Valid name was entered
            self.userNames.append(new_name)
            self.selectedUser.set(new_name)
            self.updateUserDropdownOptions()
            logger.info(f'User changed to {new_name}')
            break
