import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Initalize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseGUIApp:
    """
    This class creates a TKinter GUI to be used by the pose recorder class.
    """

    def __init__(self):
        # Initialize GUI
        self.gui = tk.Tk()
        self.gui.minsize(800, 600)
        self.gui.title('Swing Recorder')
        self.gui.configure(bg='#1e1e1e')
        self.guiStyle = ttk.Style(self.gui)
        self.guiStyle.theme_use('clam')

        # Create dropdown lists
        self.dropdownFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.userOptions = ['New user']
        self.selectedOption = tk.StringVar()
        self.selectedOption.set(self.userOptions[0])
        self.dropdown = ttk.OptionMenu(self.gui, self.selectedOption, self.userOptions[0], *self.userOptions)
        self.dropdown.pack(padx=10, pady=10)

        # Node selection dropdown (to be populated in PoseRecorderApp)
        self.selectedNode = tk.StringVar()

        # Add Analyze button to the top right
        self.analyzeButton = tk.Button(self.gui, text='Analyze', font=('Arial', 12), command=self.show_analysis_frame)
        self.analyzeButton.place(relx=1.0, x=-10, y=10, anchor='ne')

        # Frame to hold analysis plot
        self.plotFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.plotFrame.pack(fill='both', expand=False, padx=10, pady=10)

        # Create video frame in top row
        self.videoFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.videoFrame.pack(side='top', fill='both', expand=True)
        self.videoFrame.pack_propagate(False)
        self.videoSection = tk.Label(self.videoFrame)
        self.videoSection.pack(padx=10, pady=10, fill='both', expand=True)

        # Analysis frame which shows up when the analyze button is clicked
        self.analysisFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.analysisNodeVar = tk.StringVar()
        self.analysisNodeDropdown = ttk.OptionMenu(self.analysisFrame, self.analysisNodeVar, '')
        self.analysisNodeDropdown.pack(padx=10, pady=10)
        self.analysisFrame.pack_forget()

        # Create text frame in middle row
        self.textFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.textFrameText = tk.Label(self.textFrame, text='Select an option:', font=('Arial', 14), fg='white', bg='#1e1e1e')
        self.textFrame.pack(fill='x', pady=0)
        self.textFrameText.pack()

        # Create button frame in bottom row
        self.buttonFrame = tk.Frame(self.gui, height=100, bg='#1e1e1e')
        self.buttonFrame.pack(side='bottom', fill='x', pady=10)
        self.buttonFrame.pack_propagate(False)
        self.btnRecord = tk.Button(self.buttonFrame, text='Record 5 Seconds', font=('Arial', 14), width=20)
        self.btnRecord.pack(side='left', expand=True, padx=10)
        self.btnGood = tk.Button(self.buttonFrame, text='Good', font=('Arial', 14), width=20)
        self.btnBad = tk.Button(self.buttonFrame, text='Bad', font=('Arial', 14), width=20)
        self.countdownText = tk.Label(self.buttonFrame, font=('Arial', 14), fg='white', bg='#1e1e1e')

        # Add tracing to selected user
        self.selectedOption.trace_add('write', self.handleSelection)


    def updateDropdownOptions(self):
        """
        Updates the options in the OptionMenu widget.
        """
        menu = self.dropdown['menu']
        menu.delete(0, 'end')
        for option in self.userOptions:
            menu.add_radiobutton(label=option, variable=self.selectedOption, value=option)
        current_selection = self.selectedOption.get()
        if current_selection in self.userOptions:
            pass
        else:
            self.selectedOption.set(self.userOptions[0])


    def handleSelection(self, *args):
        """
        Handles the selection of a user from the dropdown menu and sets as selected user.
        """
        selected_value = self.selectedOption.get()
        if selected_value == 'New user':
            while True:
                new_name = simpledialog.askstring('New User', "Enter the new user's name:", parent=self.gui)

                if new_name is None:
                    logger.info('New user entry cancelled.')
                    if len(self.userOptions) > 1:
                        self.selectedOption.set(self.userOptions[0])
                    break

                new_name = new_name.strip()
                if not new_name:
                    logger.info('Empty name entered.')
                    messagebox.showerror('Error', 'No name was entered.')
                    continue

                if new_name == 'New user':
                    logger.info('New user was entered as name')
                    messagebox.showerror('Error', 'Invalid user name entered')
                    continue

                if new_name in self.userOptions:
                    logger.info(f"User '{new_name}' already exists.")
                    messagebox.showerror('Error', 'User already exists, please enter a different name.')
                    continue

                # Valid, unique name
                new_user_index = self.userOptions.index('New user')
                self.userOptions.insert(new_user_index, new_name)
                self.selectedOption.set(new_name)
                self.updateDropdownOptions()
                break
