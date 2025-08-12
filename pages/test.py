from tkinter import Tk, Canvas
from PIL import Image, ImageTk

root = Tk()

pipe_canvas = Canvas(root, width=400, height=400)
pipe_canvas.pack()

# Load your image using PIL and convert it to a format Tkinter can use
image_path = "pipeline_schema/2413446.png"
image = Image.open(image_path)
icon = ImageTk.PhotoImage(image)

# Keep a reference to the image to avoid garbage collection
icons = []  # Store references to images here
icons.append(icon)

# Example coordinates for bend_x and bend_y
bend_x, bend_y = 100, 100

# Draw the image on the canvas
pipe_canvas.create_image(bend_x, bend_y, image=icon)

root.mainloop()
