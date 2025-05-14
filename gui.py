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

        # Create menu bar
        self.menubar = tk.Menu(self.gui)
        self.gui.config(menu=self.menubar)

        # Create File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New User", command=self.addNewUser)
        self.file_menu.add_command(label="Change User", command=self.show_user_selection)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.gui.quit)
        
        # Create View menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Analysis", command=self.show_analysis_frame)
        self.view_menu.add_command(label="Home", command=self.show_home_frame)

        # Create middle frame for video and plot
        self.middleFrame = tk.Frame(self.gui, bg='#1e1e1e', height=500)
        self.middleFrame.pack(side='top', fill='both', expand=True, padx=10, pady=10)
        self.middleFrame.pack_propagate(False)

        # Create bottom frame for text and buttons
        self.bottomFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.bottomFrame.pack(side='bottom', fill='x')

        # Add profile label at top
        self.userNames = []
        self.selectedUser = tk.StringVar()
        self.profileFrame = tk.Frame(self.gui, bg='#1e1e1e', height=30)
        self.profileFrame.pack(side='top', fill='x', padx=10, pady=(5,0))
        self.profileLabel = tk.Label(self.profileFrame, text='Profile: None', font=('Arial', 12), fg='white', bg='#1e1e1e')
        self.profileLabel.pack(side='left')

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


    def show_user_selection(self):
        """
        Show dialog to select user
        """
        if not self.userNames:
            messagebox.showinfo('No Users', 'Please create a new user first.')
            return

        # Create a new window for user selection
        dialog = tk.Toplevel(self.gui)
        dialog.title('Select User')
        dialog.geometry('300x400')
        dialog.configure(bg='#1e1e1e')
        
        # Make dialog modal
        dialog.transient(self.gui)
        dialog.grab_set()
        
        # Add listbox for users
        listbox = tk.Listbox(dialog, font=('Arial', 12), bg='#2d2d2d', fg='white', selectmode='single')
        listbox.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add users to listbox
        for user in self.userNames:
            listbox.insert(tk.END, user)
            
        # Add select button
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_user = listbox.get(selection[0])
                self.selectedUser.set(selected_user)
                self.update_profile_label()
                dialog.destroy()
                
        select_btn = tk.Button(dialog, text='Select', command=on_select, font=('Arial', 12))
        select_btn.pack(pady=10)
        
        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

    def update_profile_label(self):
        """
        Update the profile label with current user
        """
        current_user = self.selectedUser.get()
        self.profileLabel.config(text=f'Profile: {current_user if current_user else "None"}')

    def updateUserDropdownOptions(self):
        """
        Updates the user names in the dropdown menu
        """
        if self.userNames:
            if not self.selectedUser.get():
                self.selectedUser.set(self.userNames[0])
            self.update_profile_label()

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
            logger.info(f'Added new user {new_name}')
            break
