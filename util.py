import os
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace

def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )
    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img_path, db_path):
    # Use DeepFace to find the best match
    try:
        result = DeepFace.find(img_path=img_path, 
                               db_path=db_path, 
                               model_name='Facenet512', 
                               detector_backend='mtcnn')
        
        if result.empty:
            return 'unknown_person'
        
        # Get the best match (the first result)
        best_match = result.iloc[0]['identity']
        person_name = os.path.basename(best_match).replace(".pkl", "")
        
        return person_name
    except Exception as e:
        return f"Error: {str(e)}"
