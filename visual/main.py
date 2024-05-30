import cv2
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import tkinter as tk
from tkinter import ttk, Text, messagebox

# Initialize the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

class VQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Question Answering App")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        self.style = ttk.Style()
        self.style.configure("TFrame", background="#2c3e50")
        self.style.configure("TLabelFrame", background="#34495e", foreground="#ecf0f1", font=("Helvetica", 12, "bold"))
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), background="#1abc9c", foreground="#000000")
        self.style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=("Helvetica", 10))
        self.style.configure("TText", background="#34495e", foreground="#ecf0f1", font=("Helvetica", 10))

        self.canvas = tk.Canvas(root, bg="#2c3e50", bd=0, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.create_widgets()
        self.raw_image = None

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def create_widgets(self):
        # Frame for capturing the image
        capture_frame = ttk.LabelFrame(self.inner_frame, text="Capture Image", padding=10)
        capture_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.capture_button = ttk.Button(capture_frame, text="Capture", command=self.capture_image)
        self.capture_button.pack(side="left", padx=10, pady=10)

        self.image_label = ttk.Label(capture_frame, text="No image captured", anchor="center", background="#34495e", foreground="#ecf0f1")
        self.image_label.pack(side="left", padx=10, pady=10)

        # Frame for questions and answers
        qa_frame = ttk.LabelFrame(self.inner_frame, text="Questions and Answers", padding=10)
        qa_frame.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        questions_label = ttk.Label(qa_frame, text="Enter your questions (one per line):")
        questions_label.pack(anchor="center", padx=10, pady=(10, 0))

        self.questions_text = Text(qa_frame, width=60, height=5, font=("Helvetica", 10), bg="#34495e", fg="#ecf0f1", insertbackground="#ecf0f1")
        self.questions_text.pack(padx=10, pady=5)

        self.submit_button = ttk.Button(qa_frame, text="Submit", command=self.answer_questions)
        self.submit_button.pack(pady=10)

        answers_label = ttk.Label(qa_frame, text="Answers:")
        answers_label.pack(anchor="center", padx=10, pady=(10, 0))

        self.answers_text = Text(qa_frame, width=60, height=10, state="disabled", font=("Helvetica", 10), bg="#34495e", fg="#ecf0f1", wrap="word")
        self.answers_text.pack(padx=10, pady=5)

        # Center-aligning the inner_frame
        self.inner_frame.pack_propagate(False)
        self.inner_frame.update_idletasks()
        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.create_window((self.canvas.winfo_width() // 2, 0), window=self.inner_frame, anchor="n")

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read frame")
            cap.release()
            return

        cap.release()

        # Convert the image from BGR (OpenCV format) to RGB (PIL format)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.raw_image = Image.fromarray(image)
        self.display_image(self.raw_image)

    def display_image(self, image):
        img_tk = ImageTk.PhotoImage(image=image)
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk, text="")

    def answer_questions(self):
        if self.raw_image is None:
            messagebox.showerror("Error", "No image captured")
            return

        questions = self.questions_text.get("1.0", tk.END).strip().split("\n")
        if not questions:
            messagebox.showerror("Error", "No questions provided")
            return

        answers = []
        for question in questions:
            if not question.strip():
                continue
            try:
                inputs = processor(self.raw_image, question, return_tensors="pt")
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                answers.append(f"Q: {question}\nA: {answer}")
            except Exception as e:
                answers.append(f"Q: {question}\nA: [Could not generate an answer: {str(e)}]")

        self.display_answers(answers)

    def display_answers(self, answers):
        self.answers_text.configure(state="normal")
        self.answers_text.delete("1.0", tk.END)
        self.answers_text.insert(tk.END, "\n\n".join(answers))
        self.answers_text.configure(state="disabled")

# Create the main window
root = tk.Tk()
app = VQAApp(root)
root.mainloop()
