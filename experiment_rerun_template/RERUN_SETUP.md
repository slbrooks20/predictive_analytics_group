# Rerun CLI Setup

This project uses four separate command-line interfaces (CLIs) for rerunning the benchmark:

- Claude Code
- Gemini CLI
- Codex CLI
- Mistral Vibe

This file covers CLI installation, authentication, and the repo bootstrap step required before using the task prompts.

## macOS / Linux / WSL

Run the bundled installer from the repo root:

```bash
./install_clis.sh
```

If you want to install them manually, these are the commands used by the script:

```bash
npm install -g @google/gemini-cli
curl -fsSL https://claude.ai/install.sh | bash
npm i -g @openai/codex
curl -LsSf https://mistral.ai/vibe/install.sh | bash
```

## Windows

### Claude Code

PowerShell:

```powershell
irm https://claude.ai/install.ps1 | iex
```

CMD:

```cmd
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

### Gemini CLI

```powershell
npm install -g @google/gemini-cli
```

### Codex CLI

```powershell
npm i -g @openai/codex
```

### Mistral Vibe

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" && uv tool install mistral-vibe
```

## After Installation

Authenticate each CLI you plan to use before rerunning the experiments.

Typical commands are:

```bash
claude
gemini
codex
vibe --setup
```

Then install the Python dependencies for the rerun environment:

```bash
python3 -m pip install -r requirements.txt
```

## Before running the task prompts

From the repo root, bootstrap the per-model project folders:

```bash
./prepare_rerun_template.sh
```

That script creates `experiment_rerun_template/<model>/project/` with:

- `dataset/hour.csv`
- `scripts/`
- `outputs/`
- `requirements.txt`
- `run_pipeline.py`

Then verify all of the following:

- The target model folder under `experiment_rerun_template/` now contains a `project/` subfolder with the expected files and output subdirectories.
- You are launching the correct CLI for that model family. The shared task prompt files are benchmark instructions, not CLI-specific launcher commands.
- You `cd` into `experiment_rerun_template/<model>/project` before pasting the prompts.
