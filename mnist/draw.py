import tkinter as tk
import numpy as np
import struct

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 Drawing App")
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.pixel_data = np.zeros((28, 28), dtype=np.float32)
        self.cell_size = 10

        self.save_button = tk.Button(self.root, text="Save", command=self.save_drawing)
        self.save_button.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_rectangle((x // self.cell_size) * self.cell_size,
                                     (y // self.cell_size) * self.cell_size,
                                     (x // self.cell_size) * self.cell_size + self.cell_size,
                                     (y // self.cell_size) * self.cell_size + self.cell_size,
                                     fill="black")
        self.update_pixel_data(x, y)

    def update_pixel_data(self, x, y):
        grid_x, grid_y = x // self.cell_size, y // self.cell_size
        if 0 <= grid_x < 28 and 0 <= grid_y < 28:
            self.pixel_data[grid_y, grid_x] = 1.0

    def save_drawing(self):
        with open("drawing.bin", "wb") as f:
            for row in self.pixel_data:
                for pixel in row:
                    f.write(struct.pack('f', pixel))
        print("Drawing saved to drawing.bin")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

