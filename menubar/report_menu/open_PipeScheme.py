# def open_PipeScheme(self):
#     print("pipetally path inside open_pipescheme", self.pipetally_dir)
#     try:
#         import subprocess, sys, os
#         pipeline_path = os.path.join("pipeline_schema", "pipeline_schema.py")
#         subprocess.Popen([sys.executable, pipeline_path, self.project_root])
#     except Exception as e:
#         self.open_Error(f"Error running Pipeline Schema:\n{e}")


def open_PipeScheme(self):
    print("pipetally path:", self.pipetally_dir)

    try:
        from pipeline_schema.pipeline_schema import run_pipe_schema

        run_pipe_schema(pipe_tally=self.pipetally_dir)  # ✅ no parent

    except Exception as e:
        self.open_Error(f"Error running Pipeline Schema:\n{e}")