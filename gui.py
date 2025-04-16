import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Initalize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseGUIApp:
    """
    TODO: FILL THIS OUT
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
        self.user_options = ['New user']
        self.selected_option = tk.StringVar()
        self.selected_option.set(self.user_options[0])
        self.dropdown = ttk.OptionMenu(self.gui, self.selected_option, self.user_options[0], *self.user_options, command=self.handle_selection)
        self.dropdown.pack(padx=10, pady=10)

        # Create video frame in top row
        self.videoFrame = tk.Frame(self.gui, bg='#1e1e1e')
        self.videoFrame.pack(side='top', fill='both', expand=True)
        self.videoFrame.pack_propagate(False)
        self.videoSection = tk.Label(self.videoFrame)
        self.videoSection.pack(padx=10, pady=10, fill='both', expand=True)

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

        self.gui.mainloop()


    def update_dropdown_options(self):
        """Updates the options in the OptionMenu widget."""
        menu = self.dropdown["menu"]
        menu.delete(0, "end")
        for option in self.user_options:
            menu.add_command(label=option, command=lambda value=option: self.selected_option.set(value))
        current_selection = self.selected_option.get()
        if current_selection in self.user_options:
            # Find the index if needed, but setting the variable usually suffices
            pass  # The trace should handle the visual update via set()
        elif self.user_options:
            # If current selection is invalid, reset to the first option
            self.selected_option.set(self.user_options[0])
        else:
            # Handle case where options list is empty (edge case)
            self.selected_option.set("")  # Or some default placeholder


    def handle_selection(self, selected_value):
        """
        Callback function when an option is selected.
        """
        if selected_value == 'New user':
            new_name = simpledialog.askstring('New User', "Enter the new user's name:", parent=self.gui)

            if new_name:
                new_name = new_name.strip()
                if new_name and new_name not in self.user_options:
                    new_user_index = self.user_options.index('New user')
                    self.user_options.insert(new_user_index, new_name)
                    self.update_dropdown_options()
                    self.selected_option.set(new_name)
                elif new_name in self.user_options:
                    self.selected_option.set(new_name)
                    print(f"User '{new_name}' already exists.")
                else:
                    logger.info('Invalid name entered.')
                    if len(self.user_options) > 1:
                        self.selected_option.set(self.user_options[0])
                    else:
                        pass

            else:
                if len(self.user_options) > 1:
                    self.selected_option.set(self.user_options[0])
                logger.info('New user entry cancelled.')
