import tkinter as tk
from tkinter import filedialog
import pickle
import numpy as np
from scipy.sparse import csr_matrix

CELL_SIZE = 30

class MatrixEditor:
    def __init__(self, root, rows=10, cols=10):
        self.root = root
        self.rows = rows
        self.cols = cols

        self.data = np.zeros((self.rows, self.cols))

        # Canvas drawing surface
        self.canvas = tk.Canvas(root, width=self.cols * CELL_SIZE,
                                height=self.rows * CELL_SIZE)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Resize", command=self.resize_matrix).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save as Pickle", command=self.save_pickle).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Load Pickle", command=self.load_pickle).pack(side=tk.LEFT)

        self.draw()

    # --- Drawing ----
    def draw(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                value = self.data[r, c]
                color = "white" if value == 0 else "black"
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    # --- Edit with clicks ---
    def left_click(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r, c] = 1
        self.draw()

    def right_click(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r, c] = 0
        self.draw()

    # --- Resize matrix ---
    def resize_matrix(self):
        win = tk.Toplevel(self.root)
        win.title("Resize")

        tk.Label(win, text="Rows:").grid(row=0, column=0)
        tk.Label(win, text="Cols:").grid(row=1, column=0)

        row_entry = tk.Entry(win)
        col_entry = tk.Entry(win)

        row_entry.insert(0, str(self.rows))
        col_entry.insert(0, str(self.cols))

        row_entry.grid(row=0, column=1)
        col_entry.grid(row=1, column=1)

        def apply_resize():
            new_r = int(row_entry.get())
            new_c = int(col_entry.get())
            new_mat = np.zeros((new_r, new_c))
            rows_to_copy = min(new_r, self.rows)
            cols_to_copy = min(new_c, self.cols)
            new_mat[:rows_to_copy, :cols_to_copy] = self.data[:rows_to_copy, :cols_to_copy]
            self.rows, self.cols = new_r, new_c
            self.data = new_mat
            self.canvas.config(width=new_c * CELL_SIZE, height=new_r * CELL_SIZE)
            self.draw()
            win.destroy()

        tk.Button(win, text="Apply", command=apply_resize).grid(row=2, column=0, columnspan=2)

    # --- Save ---
    def save_pickle(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pkl")
        if not filename:
            return
        csr = csr_matrix(self.data)
        with open(filename, "wb") as f:
            pickle.dump(csr, f)
        print("Saved to", filename)

    # --- Load ---
    def load_pickle(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if not filename:
            return
        with open(filename, "rb") as f:
            csr = pickle.load(f)

        dense = csr.toarray()
        self.rows, self.cols = dense.shape
        self.data = dense

        self.canvas.config(width=self.cols * CELL_SIZE,
                           height=self.rows * CELL_SIZE)
        self.draw()
        print("Loaded", filename)


# Run the GUI
root = tk.Tk()
root.title("CSR Matrix Editor")
MatrixEditor(root, rows=12, cols=16)
root.mainloop()
