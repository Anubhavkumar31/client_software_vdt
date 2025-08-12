import tkinter as tk
from tkinter import *
from tkinter import ttk 
from PIL import ImageTk, Image, ImageGrab
import mysql.connector
import pandas as pd
from PIL import Image, ImageTk
from tkinter import Tk, Canvas
from fpdf import FPDF
from utils import resource_path
import hashlib
# Setup MySQL connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='mfldesktop1'
)
cursor = conn.cursor(buffered=True)

# Create the main window
root = tk.Tk()
root.title("Pipeline Scheme Report and Pipe Number Visualizer")
# root.iconbitmap('pipeline_schema/LOGO-withoutbg.ico')
root.iconbitmap(resource_path('pipeline_schema/LOGO-withoutbg.ico'))
root.geometry("500x500+750+50")

# Setup canvas and scrollbar for the main window
main_canvas  = tk.Canvas(root, bg="white")
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
main_canvas.configure(yscrollcommand=scrollbar.set)


frame = ttk.Frame(main_canvas, width=1250, height=1540, style='My.TFrame')
frame.pack_propagate(False)

# Create a style for the frame
style = ttk.Style()
style.configure('My.TFrame', background='white')

main_canvas.create_window((0, 0), window=frame, anchor="nw")
main_canvas.create_window((0, 0), window=frame, anchor="nw")
main_canvas.config(scrollregion=(0,0,300,1600))

def update_scrollregion(event):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

frame.bind('<Configure>', update_scrollregion)

# Add logo to the page
# img = Image.open('pipeline_schema/LOGO-withoutbg.png').convert("RGBA")
img_path = resource_path('pipeline_schema/LOGO-withoutbg.png')
img = Image.open(img_path).convert("RGBA")

icon_path = resource_path('pipeline_schema/LOGO-withoutbg.ico')
root.iconbitmap(icon_path)
resized_img = img.resize((200, 100))
white_bg = Image.new("RGBA", resized_img.size, "WHITE")
composite = Image.alpha_composite(white_bg, resized_img)
# img = ImageTk.PhotoImage(composite)
# img_label = tk.Label(frame, image=img, bg="white")
# img_label.pack(padx=(1000, 80))
img = ImageTk.PhotoImage(composite)
img_label = tk.Label(frame, image=img, bg="white")
img_label.image = img  # <-- prevent garbage collection
img_label.pack(padx=(1000, 80))


# Data entry section for pipeline details
run_id_entry = tk.Entry(frame, bd=1, width=35)

run_id = Label(frame, text='Run Id:', background='white')
run_id.config(font=('verdana', 12))
client = Label(frame, text='Client:', background='white')
client.config(font=('verdana', 12))
pipeline_name = Label(frame, text='Pipeline name:', background='white')
pipeline_name.config(font=('verdana', 12))
report_date = Label(frame, text='Report date:', background='white')
report_date.config(font=('verdana', 12))

run_id.place(x=50, y=20)
client.place(x=50, y=50)
pipeline_name.place(x=50, y=80)
report_date.place(x=50, y=110)

client_entry = tk.Entry(frame, width=50, bd=0, font=10)
pipeline_name_entry = tk.Entry(frame, width=50, bd=0, font=10)
report_date_entry = tk.Entry(frame, width=50, bd=0, font=10)

run_id_entry.place(x=200, y=20)
client_entry.place(x=200, y=50)
pipeline_name_entry.place(x=200, y=80)
report_date_entry.place(x=200, y=110)

def get_data():
    sear = run_id_entry.get()
    cursor.execute("SELECT Pipeline_owner FROM mfldesktop.projectdetail WHERE runid = %s", (sear,))
    data = cursor.fetchone()
    client_entry.delete(0, tk.END)
    client_entry.insert(0, str(data[0]) if data else "No data found")

    cursor.execute("SELECT Pipeline_Name FROM mfldesktop.projectdetail WHERE runid = %s", (sear,))
    data = cursor.fetchone()
    pipeline_name_entry.delete(0, tk.END)
    pipeline_name_entry.insert(0, str(data[0]) if data else "No data found")

    cursor.execute("SELECT Report_date FROM mfldesktop.projectdetail WHERE runid = %s", (sear,))
    data = cursor.fetchone()
    report_date_entry.delete(0, tk.END)
    report_date_entry.insert(0, str(data[0]) if data else "No data found")

search_button = Button(frame, text="Search", command=get_data)
search_button.place(x=430, y=17)

# Excel file reading and visualization
# excel_file = 'backend/files/datalog/pipe_tally_sheet2.xlsx'
excel_file = resource_path('backend/files/datalog/pipe_tally_sheet2.xlsx')


try:
    data = pd.read_excel(
        excel_file,
        engine='openpyxl',
        na_values=[''],
        keep_default_na=False
    )
    data.columns = data.columns.str.strip()

except Exception as e:
    print(f"Error reading Excel file: {e}")
    data = pd.DataFrame()

# Pipe number visualization section
pipe_canvas = Canvas(frame, width=1240, height=1940, bg="white")
pipe_canvas.pack()

# Define the chunking logic for grouping pipe numbers
chunks = []
for pipe_number, group in data.groupby('Pipe Number'):
    chunks.append(group)

# Logic to draw 50 chunks per page
start_x = 50
start_y = 100
rect_width = 100
rect_height = 50
space_between = 20

def display_chunks(page_number=0):
    # Clear the canvas
    pipe_canvas.delete("all")

    start_x = 50
    start_y = 100
    count = 0

    chunks_per_page = 100
    start_chunk = page_number * chunks_per_page
    end_chunk = start_chunk + chunks_per_page
    y_coordinate = 0
    for i in range(10):
        pipe_canvas.create_text(start_x, start_y + y_coordinate - 40, text=f"Pipe Number:", font=("Helvetica", 10))
        pipe_canvas.create_text(start_x, start_y + y_coordinate- 25, text=f"Length(m):", font=("Helvetica", 8))
        pipe_canvas.create_text(start_x, start_y + y_coordinate - 10, text=f"WT(mm):", font=("Helvetica", 8))
        y_coordinate += 110
    for chunk_index, chunk in enumerate(chunks[start_chunk:end_chunk]):
        if count == 10:
            start_y += 110
            start_x = 50
            count = 0

        joint_length = chunk['Pipe Length(m)'].mean()
        wall_thickness = chunk['WT (mm)'].mean()
        pipe_number = chunk['Pipe Number'].max()

        x1 = start_x + (chunk_index % 10) * (rect_width + space_between)
        y1 = start_y
        x2 = x1 + rect_width
        y2 = y1 + rect_height

        # Draw the rectangle representing the pipe
        pipe_canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="white")
        count += 1

        # Display the pipe details (joint number, avg length, avg wall thickness)
        pipe_canvas.create_text(x1 + rect_width // 2, y1 - 40, text=f"{pipe_number}", font=("Helvetica", 10))
        pipe_canvas.create_text(x1 + rect_width // 2, y1 - 25, text=f"{joint_length:.2f}", font=("Helvetica", 8))
        pipe_canvas.create_text(x1 + rect_width // 2, y1 - 10, text=f"{wall_thickness:.2f}", font=("Helvetica", 8))
        
        # Check if the pipe has corrosion in the 'Feature Identification' column
        if 'Feature Identification' in chunk.columns and ('Corrosion' in chunk['Feature Identification'].values or 'corrosion' in chunk['Feature Identification'].values):
            
            # Filter the rows where 'Feature Identification' contains 'Corrosion' or 'corrosion'
            corrosion_row = chunk[chunk['Feature Identification'].str.contains('Corrosion|corrosion', case=False)]
            
            
            # Check if both 'Distance to U/S GW(m)', 'Pipe Length(m)', and 'Orientation o clock' exist in the corrosion_row columns
            if 'Distance to U/S GW(m)' in corrosion_row.columns and 'Pipe Length(m)' in corrosion_row.columns and 'Orientation o clock' in corrosion_row.columns:
                
                # Fetch the 'Distance to U/S GW(m)' value
                distance_to_upstream = corrosion_row['Distance to U/S GW(m)'].values[0]
                
                # Fetch the 'Pipe Length(m)' value
                pipe_length = corrosion_row['Pipe Length(m)'].values[0]

                # Fetch the clock orientation (e.g., 3:00, 6:00)
                clock_orientation = corrosion_row['Orientation o clock'].values[0]
            # Check if clock_orientation is a datetime.time object
                if isinstance(clock_orientation, pd.Timestamp):
                    clock_orientation = clock_orientation.time()  # Convert to time object if it's a Timestamp
                # Extract hour and minute from the datetime.time object
                hour = clock_orientation.hour  # Get the hour part
                minute = clock_orientation.minute  # Get the minute part

                # Calculate the angle based on both hour and minute
                angle = (hour % 12 + minute / 60) * 30  # Each hour is 30 degrees (360 degrees / 12 hours)
                # Adjust corrosion_x according to the distance to upstream
                # Scale the distance to upstream as per the canvas coordinate system
                corrosion_x = x1 + (distance_to_upstream * (100 / pipe_length))  # Adjust scale_factor as necessary

                # Adjust corrosion_y based on clock orientation
                if 0 <= angle <= 180:
                    # Between 12:00 and 6:00 -> y1 to y2
                    corrosion_y = y1 + ((y2 - y1) / 180) * angle
                else:
                    # Between 6:00 and 12:00 -> y2 back to y1
                    corrosion_y = y2 - ((y2 - y1) / 180) * (angle - 180)
                
                # Define the radius of the corrosion indicator
                radius = 3  # Size of the rust point
                
                # Draw the corrosion indicator (rust color)
                #pipe_canvas.create_oval(corrosion_x - radius, corrosion_y - radius, corrosion_x + radius, corrosion_y + radius, fill="#B7410E")
        if 'Feature Identification' in chunk.columns and ('Bend' in chunk['Feature Identification'].values or 'bend' in chunk['Feature Identification'].values):
            bend_row = chunk[chunk['Feature Identification'].str.contains('Bend|bend', case=False)]        
            if not bend_row.empty and 'Distance to U/S GW(m)' in bend_row.columns and 'Pipe Length(m)' in bend_row.columns and 'Orientation o clock' in bend_row.columns:
                distance_to_upstream = bend_row['Distance to U/S GW(m)'].values[0] if not pd.isna(bend_row['Distance to U/S GW(m)'].values[0]) else None
                pipe_length = bend_row['Pipe Length(m)'].values[0] if not pd.isna(bend_row['Pipe Length(m)'].values[0]) else None
                clock_orientation = bend_row['Orientation o clock'].values[0] if not pd.isna(bend_row['Orientation o clock'].values[0]) else None

                if distance_to_upstream is not None and pipe_length is not None and clock_orientation is not None:
                    # Calculate the bend_x and bend_y as before
                    if isinstance(clock_orientation, pd.Timestamp):
                        clock_orientation = clock_orientation.time()
                    hour = clock_orientation.hour
                    minute = clock_orientation.minute
                    angle = (hour % 12 + minute / 60) * 30

                    bend_x = x1 + (distance_to_upstream * (100 / pipe_length))  # Adjust scale_factor as necessary
                    if 0 <= angle <= 180:
                        bend_y = y1 + ((y2 - y1) / 180) * angle
                    else:
                        bend_y = y2 - ((y2 - y1) / 180) * (angle - 180)

                    
                    pipe_canvas.create_text(bend_x, bend_y, text="*", font=("Arial", 20), fill="black")

                   
                    
        if 'Feature Type' in chunk.columns and ('Metal Loss' in chunk['Feature Type'].values or 'Metal loss' in chunk['Feature Type'].values or 'metal loss' in chunk['Feature Type'].values):

            chunk = chunk.dropna(subset=['Feature Type'])

            # Filter the rows where 'Feature Type' contains 'Metal Loss' or similar variations
            metal_loss_row = chunk[chunk['Feature Type'].str.contains('Metal Loss|Metal loss|metal loss', case=False)]

            # Check if metal_loss_row is not empty and all necessary columns are present
            required_columns = ['Distance to U/S GW(m)', 'Pipe Length(m)', 'Orientation o clock', 'Depth %', 'Type']
            missing_columns = [col for col in required_columns if col not in metal_loss_row.columns]

            if not metal_loss_row.empty and all(col in metal_loss_row.columns for col in required_columns):
                
                # Fetch the values
                distance_to_upstream = metal_loss_row['Distance to U/S GW(m)'].values[0]
                pipe_length = metal_loss_row['Pipe Length(m)'].values[0]
                clock_orientation = metal_loss_row['Orientation o clock'].values[0]
                depth_percent = metal_loss_row['Depth %'].values[0]
                corrosion_type = metal_loss_row['Type'].values[0]

                '''print("Distance to U/S GW:", distance_to_upstream)
                print("Pipe Length:", pipe_length)
                print("Clock Orientation:", clock_orientation)
                print("Depth%:", depth_percent)
                print("Corrosion Type:", corrosion_type)'''

                # Check if clock_orientation is a datetime.time object
                if isinstance(clock_orientation, pd.Timestamp):
                    clock_orientation = clock_orientation.time()  # Convert to time object if it's a Timestamp

                # Extract hour and minute from the datetime.time object
                hour = clock_orientation.hour  # Get the hour part
                minute = clock_orientation.minute  # Get the minute part

                # Calculate the angle based on both hour and minute
                angle = (hour % 12 + minute / 60) * 30  # Each hour is 30 degrees (360 degrees / 12 hours)
                 

                # Adjust metal_loss_x according to the distance to upstream
                # Scale the distance to upstream as per the canvas coordinate system
                metal_loss_x = x1 + (distance_to_upstream * (100 / pipe_length))  # Adjust scale_factor as necessary

                # Adjust metal_loss_y based on clock orientation
                if 0 <= angle <= 180:
                    # Between 12:00 and 6:00 -> y1 to y2
                    metal_loss_y = y1 + ((y2 - y1) / 180) * angle
                else:
                    # Between 6:00 and 12:00 -> y2 back to y1
                    metal_loss_y = y2 - ((y2 - y1) / 180) * (angle - 180)

                # Define the radius of the corrosion indicator
                radius = 3  # Size of the rust point

                # Draw the corrosion indicator based on depth and type
                if depth_percent > 50 and corrosion_type == "Internal":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, outline="red", width = 2)
                if depth_percent > 50 and corrosion_type == "External":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, fill="red")
                if 20 < depth_percent <= 50 and corrosion_type == "Internal":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, outline="blue", width = 2)
                if 20 < depth_percent <= 50 and corrosion_type == "External":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, fill="blue")
                if depth_percent <= 20 and corrosion_type == "Internal":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, outline="green", width = 2)
                if depth_percent <= 20 and corrosion_type == "External":
                    pipe_canvas.create_oval(metal_loss_x - radius, metal_loss_y - radius, metal_loss_x + radius, metal_loss_y + radius, fill="green")
            

                    
                
# Slot selection dropdown menu
def on_slot_select(event):
    selected_slot = slot_var.get()
    if selected_slot:
        selected_page = int(selected_slot.split('-')[0]) // 100
        display_chunks(selected_page)

# Create list of slot options
num_chunks = len(chunks)
slots = [f"{i+1}-{i+100}" for i in range(0, num_chunks, 100)]

slot_var = StringVar()
slot_menu = ttk.Combobox(frame, textvariable=slot_var, values=slots, state="readonly")
slot_menu.set("Select Slot")
slot_menu.place(x=500, y=17)
slot_menu.bind("<<ComboboxSelected>>", on_slot_select)

# Initial display of chunks
display_chunks(0)
def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest()

# Function to capture and save the window as a PDF
def save_as_pdf():
    # Update the window to ensure all elements are rendered
    root.update()

    # Set to store the hash of captured images to avoid duplicates
    image_hashes = set()
    images = []

    # Total height of the frame content
    total_height = frame.winfo_height()

    # Viewport height
    viewport_height = root.winfo_height()
    
    # Define the scroll steps in units
    scroll_steps = [710, 710]  # Example steps for your case

    # Calculate the normalized scroll positions
    normalized_steps = [sum(scroll_steps[:i]) / total_height for i in range(len(scroll_steps) + 1)]

    for i in range(len(normalized_steps) - 1):
        # Scroll to the current position
        main_canvas.yview_moveto(normalized_steps[i])

        # Update the window to render the scrolled content
        root.update()

        # Capture the current viewport
        image = ImageGrab.grab(include_layered_windows=True)
       
        # Hash the image and check for duplicates
        image_hash = hash_image(image)
        if image_hash not in image_hashes:
            image_hashes.add(image_hash)
            images.append(image)

    # Create a PDF and add the captured images
    pdf = FPDF()
    img_width, img_height = 210, 297  # A4 page size in mm

    for i in range(0, len(images), 2):
        pdf.add_page()
        
        for j in range(2):
            if i + j < len(images):
                img = images[i + j]
                temp_image_path = f"temp_image_{i + j}.png"
                img.save(temp_image_path)
                
                # Calculate the y position to place the image
                y_pos = (img_height / 2) * j
                pdf.image(temp_image_path, x=0, y=y_pos, w=img_width, h=img_height / 2)

    pdf.output(f"Pipe Line Scheme Report.pdf")

# Create Buttons
pdf_button = Button(frame, text="Save as PDF", command=save_as_pdf)
pdf_button.place(x=1110, y=1260)
root.mainloop()

def run_app():
    window.show()
