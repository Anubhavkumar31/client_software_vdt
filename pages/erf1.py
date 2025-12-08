import tkinter as tk
from tkinter import ttk, messagebox
import math

import os
import pandas as pd
class ERF1App:
    def __init__(self, project_root=None):

        self.root = tk.Tk()
        self.root.title("ERF Calculator")

        self.project_root = project_root
        self.od_value = self.smys_value = self.maop_value = None

        # --- Dynamically load constants from project root ---
        try:
            if not self.project_root:
                raise ValueError("No project root provided to ERF1App")

            constants_path = os.path.join(self.project_root, "constants")
            if not os.path.isdir(constants_path):
                raise FileNotFoundError(f"No 'constants' folder found in {self.project_root}")

            excel_files = [f for f in os.listdir(constants_path) if f.endswith((".xls", ".xlsx"))]
            if not excel_files:
                raise FileNotFoundError("No Excel files found in constants folder.")

            excel_path = os.path.join(constants_path, excel_files[0])
            df = pd.read_excel(excel_path)

            # Flexible column matching
            for col in df.columns:
                name = col.lower()
                if "diameter" in name or "od" in name:
                    self.od_value = float(df.iloc[0][col])
                elif "smys" in name:
                    self.smys_value = float(df.iloc[0][col])
                elif "maop" in name:
                    self.maop_value = float(df.iloc[0][col])

            print(f"✅ Loaded constants from: {excel_path}")
            print("📊 OD:", self.od_value, "SMYS:", self.smys_value, "MAOP:", self.maop_value)

        except Exception as e:
            print(f"⚠️ Error loading constants: {e}")



        # 🔹 Center and default window size (resizable enabled)
        window_width = 560
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int((screen_width / 2) - (window_width / 2))
        center_y = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.resizable(True, True)
        self.root.configure(bg="#eef2f7")

        # Style setup
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TLabelframe", background="#eef2f7")
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"), background="#eef2f7")

        style.configure(
            "Dark.TButton",
            font=("Segoe UI Semibold", 11),
            background="#2f3640",
            foreground="white",
            borderwidth=0,
            focusthickness=3,
            padding=(10, 6),
        )
        style.map(
            "Dark.TButton",
            background=[("active", "#3b3b3b"), ("pressed", "#1e272e")],
            foreground=[("active", "white")],
        )

        # Header
        header = tk.Label(
            self.root,
            text="ERF (Estimated Repair Factor) Calculator",
            font=("Segoe UI Semibold", 16, "bold"),
            bg="black",
            fg="white",
            pady=10,
        )
        header.pack(fill="x")

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # --- Input Section ---
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=15)
        input_frame.pack(fill="x", pady=15)

        fields = [
            ("Length of Defect (L):", "[mm]", "Enter defect length"),
            ("Depth of Defect (d):", "[mm]", "Enter defect depth"),
            ("Outside Diameter (D):", "[mm]", "Enter outside diameter"),
            ("Pipe Thickness (T):", "[mm]", "Enter thickness"),
            ("SMYS:", "[kg/cm²]", "Enter SMYS"),
            ("MAOP:", "[kg/cm²]", "Enter MAOP"),
        ]

        # Register validation
        vcmd = (self.root.register(self.validate_numeric_input), "%P")

        self.entries = []
        for i, (label_text, unit_text, placeholder) in enumerate(fields):
            # Label (parameter name)
            ttk.Label(input_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=6, padx=(0, 10))

            # Entry box
            e = self.PlaceholderEntry(
                input_frame,
                placeholder=placeholder,
                width=22,
                font=("Segoe UI", 11),
                validate="key",
                validatecommand=vcmd,
            )
            e.grid(row=i, column=1, pady=6)
            self.entries.append(e)

            # Unit label (SI unit shown after entry box)
            tk.Label(
                input_frame,
                text=unit_text,
                font=("Segoe UI", 10, "bold"),
                bg="#eef2f7",
                fg="#222"
            ).grid(row=i, column=2, sticky="w", padx=(10, 0))

        (
            self.length_L,
            self.depth_d,
            self.od_D,
            self.thickness_T,
            self.smys,
            self.maop,
        ) = self.entries

        if self.od_value:
            self.od_D.delete(0, tk.END)
            self.od_D.insert(0, str(self.od_value))
        if self.smys_value:
            self.smys.delete(0, tk.END)
            self.smys.insert(0, str(self.smys_value))
        if self.maop_value:
            self.maop.delete(0, tk.END)
            self.maop.insert(0, str(self.maop_value))

        for widget in (self.od_D, self.smys, self.maop):
            widget.config(state="readonly")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)

        ttk.Button(
            button_frame, text="Calculate ERF", style="Dark.TButton", command=self.calculate_erf
        ).grid(row=0, column=0, padx=15)

        ttk.Button(
            button_frame, text="Reset", style="Dark.TButton", command=self.reset_fields
        ).grid(row=0, column=1, padx=15)

        # Result Section
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding=15)
        result_frame.pack(fill="both", expand=True, pady=10)

        self.result_box = tk.Text(result_frame, height=3, wrap="word", font=("Consolas", 11))
        self.result_box.pack(fill="both", expand=True)
        self.result_box.config(state="disabled", bg="#fafbfc")

        # Highlighted ERF Display
        erf_display = ttk.Frame(main_frame)
        erf_display.pack(pady=10)
        ttk.Label(erf_display, text="Final ERF:", font=("Segoe UI", 13, "bold")).pack(side="left", padx=5)
        self.erf_value_label = tk.Label(
            erf_display, text="--", font=("Segoe UI", 16, "bold"), fg="#34495e"
        )
        self.erf_value_label.pack(side="left")

        # Footer
        tk.Label(
            self.root,
            text="© Engineering Tool | Designed for reliability",
            font=("Segoe UI", 9),
            bg="#eef2f7",
            fg="#7f8c8d",
        ).pack(pady=5)

        # Key Binding
        self.root.bind("<Return>", self.calculate_erf)

    # ------------------- PLACEHOLDER ENTRY CLASS -------------------
    class PlaceholderEntry(ttk.Entry):
        def __init__(self, master=None, placeholder="", color="grey", **kwargs):
            super().__init__(master, **kwargs)
            self.placeholder = placeholder
            self.placeholder_color = color
            self.default_fg_color = self.cget("foreground")

            self.bind("<FocusIn>", self._clear_placeholder)
            self.bind("<FocusOut>", self._add_placeholder)
            self._add_placeholder()

        def _clear_placeholder(self, e):
            if self.get() == self.placeholder:
                self.delete(0, "end")
                self.config(foreground=self.default_fg_color)

        def _add_placeholder(self, e=None):
            if not self.get():
                self.insert(0, self.placeholder)
                self.config(foreground=self.placeholder_color)

    # ------------------- VALIDATION FUNCTION -------------------
    def validate_numeric_input(self, P):
        if P == "" or P == ".":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    # ------------------- CALCULATION FUNCTION -------------------
    def calculate_erf(self, event=None):
        try:
            L = float(self.length_L.get())
            d = float(self.depth_d.get()) /1000
            D = float(self.od_D.get())
            T = float(self.thickness_T.get())
            SMYS = float(self.smys.get())
            MAOP = float(self.maop.get())

            flow_stress = 1.1 * SMYS
            z_factor = (L * L) / (D * T)
            M = math.sqrt(1 + 0.8 * z_factor)
            y = 1 - (2 / 3) * (d / T)
            z = 1 - ((2 / 3) * (d / T)) / M
            k = y / z

            Estimated_failure_stress_level_SF = flow_stress * k
            Estimated_failure_stress_level_SF = flow_stress * (1 - d / T)

            estimate_failure_pressure = (2 * Estimated_failure_stress_level_SF * T) / D
            safety_factor_SF = 1.39
            safe_operating_pressure = estimate_failure_pressure / safety_factor_SF
            ERF = MAOP / safe_operating_pressure

            self.result_box.config(state="normal")
            self.result_box.delete("1.0", tk.END)
            self.result_box.insert(tk.END, f"🔹 ERF: {ERF:.4f}\n")
            self.result_box.config(state="disabled")

            self.erf_value_label.config(text=f"{ERF:.4f}")
            if ERF <= 1.0:
                self.erf_value_label.config(foreground="#2ecc71")
            elif ERF <= 1.5:
                self.erf_value_label.config(foreground="#f1c40f")
            else:
                self.erf_value_label.config(foreground="#e74c3c")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values!")

    # ------------------- RESET FUNCTION -------------------
    def reset_fields(self):
        for entry in self.entries:
            entry.delete(0, tk.END)
            entry._add_placeholder()
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", tk.END)
        self.result_box.config(state="disabled")
        self.erf_value_label.config(text="--", fg="#34495e")

    # ------------------- RUN FUNCTION -------------------
    def run(self):
        self.root.mainloop()


# ------------------- MAIN EXECUTION -------------------
if __name__ == "__main__":
    # Only for standalone testing (not from main app)
    import os
    test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app = ERF1App(project_root=test_root)
    app.run()


