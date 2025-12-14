# gui.py
import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import sys
import os
import glob
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

PYTHON = sys.executable
MODELS_DIR = "models"


class BabyBenchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BabyBench")
        self.geometry("640x520")
        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        header = ttk.Frame(self, padding=(10, 10))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="BabyBench – Self-Touch Learning",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Train and evaluate DDPG agents with optional human feedback (DeepTAMER).",
        ).pack(anchor="w")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.train_tab = ttk.Frame(notebook, padding=10)
        self.test_tab = ttk.Frame(notebook, padding=10)

        notebook.add(self.train_tab, text="Train")
        notebook.add(self.test_tab, text="Test")

        self._build_train_tab()
        self._build_test_tab()
        self._build_console()

    # ---------------- Train ----------------

    def _toggle_feedback(self):
        if self.use_tamer.get():
            self.feedback_box.config(state="readonly")
        else:
            self.feedback_box.config(state="disabled")

    def _build_train_tab(self):
        f = self.train_tab

        ttk.Label(f, text="Config").grid(row=0, column=0, sticky="w")
        self.train_config = ttk.Entry(f)
        self.train_config.insert(0, "config_selftouch.yml")
        self.train_config.grid(row=0, column=1, sticky="ew")

        ttk.Label(f, text="Episodes").grid(row=1, column=0, sticky="w")
        self.train_episodes = ttk.Entry(f)
        self.train_episodes.insert(0, "20")
        self.train_episodes.grid(row=1, column=1, sticky="ew")

        ttk.Label(f, text="Steps").grid(row=2, column=0, sticky="w")
        self.train_steps = ttk.Entry(f)
        self.train_steps.insert(0, "1000")
        self.train_steps.grid(row=2, column=1, sticky="ew")

        self.use_tamer = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f,
            text="Use DeepTAMER",
            variable=self.use_tamer,
            command=self._toggle_feedback,
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(
            f,
            text="If enabled, human feedback is used to shape the policy during training.",
            foreground="gray",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 6))

        ttk.Label(f, text="Feedback").grid(row=5, column=0, sticky="w")
        self.feedback = tk.StringVar(value="keyboard")
        self.feedback_box = ttk.Combobox(
            f,
            textvariable=self.feedback,
            values=["keyboard", "prosody", "speech"],
            state="disabled",
        )
        self.feedback_box.grid(row=5, column=1, sticky="ew")

        ttk.Button(f, text="Train", command=self.run_train).grid(
            row=6, column=0, columnspan=2, pady=8
        )

        f.columnconfigure(1, weight=1)

    # ---------------- Test ----------------

    def _build_test_tab(self):
        f = self.test_tab

        ttk.Label(f, text="Config").grid(row=0, column=0, sticky="w")
        self.test_config = ttk.Entry(f)
        self.test_config.insert(0, "config_selftouch.yml")
        self.test_config.grid(row=0, column=1, sticky="ew")

        ttk.Label(f, text="Model").grid(row=1, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(
            f,
            textvariable=self.model_var,
            state="readonly",
            values=self._list_models(),
        )
        self.model_box.grid(row=1, column=1, sticky="ew")

        if self.model_box["values"]:
            self.model_box.current(0)

        ttk.Label(
            f,
            text="Select a trained actor network from the models/ directory.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 6))

        ttk.Label(f, text="Episodes").grid(row=3, column=0, sticky="w")
        self.test_episodes = ttk.Entry(f)
        self.test_episodes.insert(0, "10")
        self.test_episodes.grid(row=3, column=1, sticky="ew")

        ttk.Label(f, text="Steps").grid(row=4, column=0, sticky="w")
        self.test_steps = ttk.Entry(f)
        self.test_steps.insert(0, "1000")
        self.test_steps.grid(row=4, column=1, sticky="ew")

        ttk.Button(f, text="Test", command=self.run_test).grid(
            row=5, column=0, columnspan=2, pady=8
        )

        f.columnconfigure(1, weight=1)

    # ---------------- Console ----------------

    def _build_console(self):
        ttk.Label(self, text="Execution log").pack(anchor="w")
        self.console = tk.Text(self, height=10, bg="black", fg="white")
        self.console.pack(fill="both", expand=False)

    # ---------------- Helpers ----------------

    def _list_models(self):
        if not os.path.isdir(MODELS_DIR):
            return []
        return glob.glob(os.path.join(MODELS_DIR, "actor_*.pt"))

    def _run_cmd(self, cmd):
        self.console.delete("1.0", tk.END)

        def run():
            p = subprocess.Popen(
                [PYTHON, "-u", *cmd[1:]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(p.stdout.readline, ""):
                self.console.insert(tk.END, line)
                self.console.see(tk.END)

        threading.Thread(target=run, daemon=True).start()

    # ---------------- Actions ----------------

    def run_train(self):
        cmd = [
            PYTHON,
            "train.py",
            "--config",
            self.train_config.get(),
            "--episodes",
            self.train_episodes.get(),
            "--steps",
            self.train_steps.get(),
        ]

        if self.use_tamer.get():
            cmd += ["--use_tamer", "--feedback", self.feedback.get()]

        self._run_cmd(cmd)

    def run_test(self):
        if not self.model_var.get():
            return

        cmd = [
            PYTHON,
            "test.py",
            "--config",
            self.test_config.get(),
            "--model",
            self.model_var.get(),
            "--episodes",
            self.test_episodes.get(),
            "--steps",
            self.test_steps.get(),
        ]

        self._run_cmd(cmd)


if __name__ == "__main__":
    BabyBenchGUI().mainloop()
