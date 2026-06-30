# def open_PipeScheme(self):
#     print("pipetally path inside open_pipescheme", self.pipetally_dir)
#     try:
#         import subprocess, sys, os
#         pipeline_path = os.path.join("pipeline_schema", "pipeline_schema.py")
#         subprocess.Popen([sys.executable, pipeline_path, self.project_root])
#     except Exception as e:
#         self.open_Error(f"Error running Pipeline Schema:\n{e}")
from main_window.components.menubar.report_menu.pipeline_schema.pipeline_schema import run_pipe_schema


def open_PipeScheme(self):
    print("pipetally path:", self.pipetally_dir)

    try:


        run_pipe_schema(pipe_tally=self.pipetally_dir)  # ✅ no parent

    except Exception as e:
        self.open_Error(f"Error running Pipeline Schema:\n{e}")