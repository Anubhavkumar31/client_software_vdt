def open_PipeScheme(self):
    try:
        import subprocess, sys, os
        pipeline_path = os.path.join("pipeline_schema", "pipeline_schema.py")
        subprocess.Popen([sys.executable, pipeline_path, self.project_root])
    except Exception as e:
        self.open_Error(f"Error running Pipeline Schema:\n{e}")