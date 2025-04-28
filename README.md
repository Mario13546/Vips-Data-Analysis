# Vips Data Analysis

## 📦 Prerequisites

Before diving in, make sure you’ve got these installed:

- **Git**  
  - [Download here](https://git-scm.com/downloads)
- **Git LFS** (required for Mac only)  
  ```bash
  # macOS (Homebrew)
  brew install git-lfs

  # Ubuntu/Debian
  sudo apt install git-lfs

  # Then enable for your user
  git lfs install
  ```
- **Python 3.11.8**:  
  - [Download here](https://www.python.org/downloads/release/python-3118/)
- **Visual Studio Code**  
  - [Download here](https://code.visualstudio.com/download)  

---

## 🗂️ Folder Creation

1. **Pick your folder.**  
   Create a folder named vips_data_analysis wherever you like

---

## 🖥️ Setting Up VS Code

1. **Open the folder**  
   - Launch VS Code  
   - `File` → `Open Folder…` → select the `vips_data_analysis` folder
2. **Open the integrated terminal**  
   - Windows/Linux: <kbd>Ctrl</kbd>+<kbd>`</kbd>  
   - macOS:      <kbd>⌘</kbd>+<kbd>`</kbd>  
   - (If that fails, hit <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> or <kbd>⌘</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, type “Toggle Terminal” → Enter.)
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
     - `Jupyter` (Microsoft) → includes all Jupyter goodies  
   - **Optional but awesome**  
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
├── data/
│   └── MM-DD-YYYY/
│       ├── chip_a/
│       │   ├── spreadsheet1.csv
│       │   └── spreadsheet2.csv
│       └── chip_b/
│           ├── spreadsheet1.csv
│           └── spreadsheet2.csv
├── graphs/
│   └── (auto‑populated by scripts)
├── scripts/
│   └── *.ipynb
├── requirements.txt
└── README.md
```

- **Date folders** must be named like `04-17-2025` (MM-DD-YYYY).  
- **Under each date**, have `chip_a/` and/or `chip_b/` with your raw `.csv` files.

---

## ▶️ Running the Analysis

1. **Copy a template notebook**  
   ```bash
   cp scripts/04-17-2025.ipynb scripts/05-01-2025.ipynb
   ```
2. **Open it in VS Code** (double‑click in the Explorer).  
3. **Select your kernel**  
   - Click “Select Kernel” (top right of the notebook editor)  
   - Choose `venv (Python 3.11.8)`  
4. **Run everything**  
   - Hit the ▶️ “Run All” button at the top.  
   - Sit back, watch the cells execute, and let the graphs magically appear under `graphs/MM-DD-YYYY/...`.

> 💡 **Tip:** If your computer fan sounds like a jet engine, you’re doing it right.
