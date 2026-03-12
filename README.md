# Vips Data Analysis

## 📦 Prerequisites

Before diving in, make sure you’ve got these installed:

- **Git**  
  - [Download here](https://git-scm.com/downloads)
- **Git LFS** (required for Linux and Mac only)  
  ```bash
  # Ubuntu/Debian
  sudo apt install git-lfs

  # macOS (Homebrew)
  brew install git-lfs

  # Then enable for your user
  git lfs install
  ```
- **Python 3**:  
  - [Download here](https://www.python.org/downloads/)
- **Visual Studio Code**  
  - [Download here](https://code.visualstudio.com/download)  

---

## 🗂️ Folder Creation

1. **Pick your folder.**
   - Create a folder named vips_data_analysis wherever you like

---

## 🖥️ Setting Up VS Code

1. **Open the folder**  
   - Launch VS Code  
   - `File` → `Open Folder…` → select the `vips_data_analysis` folder
2. **Open the integrated terminal**  
   - Windows/Linux: <kbd>Ctrl</kbd>+<kbd>`</kbd>  
   - macOS:      <kbd>⌘</kbd>+<kbd>`</kbd>  
> 💡 **Tip:** If that fails, hit <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> or <kbd>⌘</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, then type “Toggle Terminal” → Enter.
3. **Clone the repo.**
   ```bash
   git clone https://github.com/Mario13546/Vips-Data-Analysis .
   ```
4. **Set your Git identity.**
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```
5. **Install these extensions**  
   - Click the Extensions icon in the sidebar  
   - **Required**  
     - `Python` (Microsoft) → gives you Pylance & Debugger  
     - `isort`   (Microsoft)  
   - **Optional but helpful**  
     - `Rainbow CSV`    (mechatroner) — colorize your CSVs  
     - `Data Wrangler`  (Microsoft)   — visualize tables in a snap  
     - `Python Environments` (Microsoft, beta) — manage venvs  

---

## 🐍 Python Environment Setup

1. **Create a virtual environment**  
   ```bash
   # macOS/Linux
   python3 -m venv venv

   # Windows PowerShell
   python -m venv venv
   ```
2. **Activate it**  
   ```bash
   # macOS/Linux
   source venv/bin/activate

   # Windows PowerShell
   ./venv/Scripts/Activate.ps1
   ```
   You should now see `(venv)` at the start of your prompt. 🎉
3. **Verify you’re using the right pip**  
   ```bash
   pip -V
   ```
   *(Compare this to your system pip to make sure you’re isolated.)*
4. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   *(Grab a coffee—this can take a moment.)*

---

## 🗂️ Project Structure & Data Layout

```text
vips_data_analysis/
├── .venv
│   └── (all the files needed for the virtual environment)
├── analysis/
│   └── (auto‑populated by scripts)
├── data/
│   └── YYYY-MM-DD/
│       └── chip_#/
│           ├── spreadsheet_1.csv
│           └── spreadsheet_2.csv
├── graphs/
│   └── (auto‑populated by scripts)
├── script_templates
│   └── (template files for the different types of chips and tests)
├── scripts/
│   └── *.py
├── vips_data_analysis/
│   └── (package files contained here)
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

- **Date folders** must be named like `2025-04-17` (YYYY-MM-DD).  
- **Under each date**, have `chip_1/` and/or `chip_2/` with your raw `.csv` files.

---

## ▶️ Running the Analysis

1. **Install the VIPS Package**
   ```bash
   pip install -e .
   ```
2. **Copy a template notebook**  
   ```bash
   cp script_templates/single_diode_push_pull scripts/YYYY-MM-DD.ipynb
   ```
3. **Open it in VS Code** (double‑click in the Explorer).
4. **Change the Date**  
   - Change the date variable to the current date (make sure to match the convention).
> ❗ **IMPORTANT:** Ensure the date is changed for relevant analysis!
5. **Run everything**  
   - Hit the ▶️ “Run Python File” button in the top right.  
   - Sit back, watch the program execute, and let the graphs magically appear under `graphs/YYYY-MM-DD/...`.

> 💡 **Tip:** If your computer fan sounds like a jet engine, you’re doing it right.
