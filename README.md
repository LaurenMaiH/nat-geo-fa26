# [DSS x Pristine Seas] Benthic Species Clusterer
Species classification for deep sea footage.

A dark-themed web app that runs a computer-vision pipeline (detection → tracking → embeddings → clustering) over deep-sea video pulled from Dropbox, then lets you review and export the results.

---

## 🛠️ Setup

Works on **Windows 10/11** and **macOS**. The setup is the same on both — only a few commands differ, and those are shown side by side. No Python, ffmpeg, or machine-learning setup is required: Docker handles all of it.

Follow the four steps in order. Budget ~20 minutes the first time (most of it is Docker downloading in the background).

> **New to the terminal?** You'll type commands into one app:
> - **Windows:** press `Start`, type **PowerShell**, open it.
> - **macOS:** press `Cmd+Space`, type **Terminal**, open it.
>
> Copy a command, paste it in (right-click to paste in PowerShell; `Cmd+V` in Terminal), press `Enter`.

### Step 0 — Install the two required tools

| Tool | Why | Install |
|---|---|---|
| **Git** | To download the code | **Windows:** [git-scm.com/download/win](https://git-scm.com/download/win) (accept all defaults). **macOS:** run `git --version` — if it's missing, macOS will offer to install it. |
| **Docker Desktop** | Runs the app in a self-contained box | Download for your OS from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/), install, then **launch it**. On Windows, accept the WSL 2 prompt if asked. |

**Before continuing, start Docker Desktop and wait until the whale icon stops animating** (menu bar on macOS, system tray on Windows). Docker must be running for every command below. Verify with:
```
docker --version
```
If that prints a version number, you're ready.

### Step 1 — Download the code

In your terminal, go to the folder where you want the project to live, then clone it. Replace `<repo-url>` with the HTTPS or SSH URL from the repository's green **Code** button on GitHub.

```
git clone <repo-url>
cd nat-geo-fa26
```

> **Already have the repo?** Skip cloning — just `cd` into the folder and grab the latest changes:
> ```
> git pull origin main
> ```

Everything from here on is run from **inside this project folder** (the one containing `docker-compose.yml`).

### Step 2 — Add your Dropbox credentials (`.env`)

The app reads mission videos from Dropbox, so it needs three secret values. First, copy the template:

| | Command |
|---|---|
| **Windows (PowerShell)** | `copy .env.example .env` |
| **macOS** | `cp .env.example .env` |

Then open the new `.env` file in a text editor (**Notepad** on Windows, **TextEdit** or VS Code on macOS) and fill in the three values:
```
DROPBOX_APP_KEY=your_app_key_here
DROPBOX_APP_SECRET=your_app_secret_here
DROPBOX_REFRESH_TOKEN=your_refresh_token_here
```

<details>
<summary><b>Don't have these yet? Click to expand — one-time setup for your own Dropbox account.</b></summary>

1. Go to the [Dropbox App Console](https://www.dropbox.com/developers/apps) and click **Create app**.
2. Choose **Scoped access**, then **Full Dropbox** access (not "App folder" — the app needs to read mission video folders wherever they live, not just inside a dedicated app folder). Name the app and create it.
3. On the app's **Permissions** tab, enable `files.metadata.read` and `files.content.read`, then click **Submit**. The app only ever reads videos — no write scopes are needed.
4. On the app's **Settings** tab, copy the **App key** and **App secret** into `.env` as `DROPBOX_APP_KEY` / `DROPBOX_APP_SECRET`.
5. Get the refresh token (once per app). Visit this URL in your browser, swapping in your App key:
   ```
   https://www.dropbox.com/oauth2/authorize?client_id=YOUR_APP_KEY&response_type=code&token_access_type=offline
   ```
   Click **Allow**, then copy the authorization code it shows you.
6. Exchange that code for a refresh token by running this **one-line** command, filling in your values:

   | | Command |
   |---|---|
   | **Windows (PowerShell)** | `curl.exe https://api.dropbox.com/oauth2/token -d code=THE_CODE -d grant_type=authorization_code -d client_id=YOUR_APP_KEY -d client_secret=YOUR_APP_SECRET` |
   | **macOS** | `curl https://api.dropbox.com/oauth2/token -d code=THE_CODE -d grant_type=authorization_code -d client_id=YOUR_APP_KEY -d client_secret=YOUR_APP_SECRET` |

   > ⚠️ **Windows:** use `curl.exe`, not `curl`. In PowerShell, plain `curl` is a different, incompatible command and will fail.

   The JSON response includes a `refresh_token` field — that's your `DROPBOX_REFRESH_TOKEN`. It doesn't expire, so this is a one-time step.

</details>

### Step 3 — Add the sample mission data

This gives the app a deployment and mission to open by default.

1. Download the `DOEX0096_Palau` folder from [this Google Drive link](https://drive.google.com/drive/folders/1BD5iZZIn2RrYc0OxrK8cHqs3ObZ5Eu95?usp=sharing). It contains `deployment.json` and the `PLW_dscm_11` mission (crops, proxy videos, detections CSV, embeddings, and `state.json`).
2. Inside the project folder, make sure there's a folder named `processed_missions` (create it if missing).
3. Place the downloaded `DOEX0096_Palau` folder inside it, so **both** of these paths exist:
   ```
   processed_missions/DOEX0096_Palau/deployment.json
   processed_missions/DOEX0096_Palau/PLW_dscm_11/
   ```

> A **deployment** is one Dropbox root you pull videos from (e.g. one client's account, or one expedition). Each mission lives in its own folder under `processed_missions/DOEX0096_Palau/`. To add more missions to this deployment, download or process them the same way. See [Connecting a different deployment](#connecting-a-different-deployment) for adding a separate one.

### Step 4 — Build and run

With Docker Desktop running, from inside the project folder:

```
docker compose build
```
*(First time only, or after code changes. This downloads and compiles everything — it can take several minutes.)*

Then start the app:
```
docker compose up
```

Open **http://localhost:5001** in your browser. You should see the dark-themed interface with a **Pipeline** tab and a **Cluster Review** tab.

To stop the app: press `Ctrl+C` in the terminal, then run `docker compose down`.

---

<details>
<summary><b>Alternative: run without Docker (advanced)</b></summary>

If you prefer not to use Docker, you can run the app directly with Python 3.12+ and [ffmpeg](https://ffmpeg.org/) installed.

```
python -m venv venv
```
Activate it:

| | Command |
|---|---|
| **Windows (PowerShell)** | `venv\Scripts\Activate.ps1` |
| **macOS** | `source venv/bin/activate` |

Then install dependencies and run:
```
pip install -r requirements.txt
python application/app.py
```
This serves the app on **http://localhost:5000** (note: **5000**, not 5001 — the 5001 mapping is Docker-only).

> **macOS note:** port 5000 is used by the built-in *AirPlay Receiver*. If the app won't start, either turn AirPlay Receiver off in **System Settings → General → AirDrop & Handoff**, or run on another port: `PORT=5001 python application/app.py`.

Unlike Docker, this server does **not** auto-reload — stop it (`Ctrl+C`) and start it again after any code change.

</details>

---

## Connecting a different deployment

A "deployment" is a Dropbox root folder you're pulling mission videos from — e.g. a different client, or a different expedition. Each deployment keeps its own missions, crops, and clustering results completely separate.

- **Switching between deployments already set up:** use the **Deployment** dropdown in the app's top bar. Everything below it — missions, review, export — scopes to whichever deployment is selected.
- **Adding a brand-new deployment:** go to the **Pipeline** tab, and in the Deployment dropdown pick **"+ New deployment…"**. Give it a name and the Dropbox root path (e.g. `/SomeClientFolder`), fill in Mission ID as usual, and start the pipeline. The app creates the deployment's folder automatically the first time it runs — no manual setup needed.

## Exporting results

In the **Cluster Review** tab, once a mission is initialized, two buttons in the header download the clustering results as CSV:

**Export tracks** — one row per track (every animal detected and tracked, whether it ended up in a named cluster, left unassigned, or was trashed):

| Column | Meaning |
|---|---|
| `mission_id`, `video`, `track_id` | Which mission/video/track this row is |
| `status` | `clustered`, `unassigned`, or `trash` |
| `cluster_id`, `cluster_name` | Which cluster it's in (if any) |
| `n_detections` | How many frames this track was detected in |
| `first_frame`, `last_frame`, `duration_frames` | Where in the video it appears |
| `top_class` | Most common raw detector label for this track (e.g. "fish") |
| `max_confidence`, `mean_confidence` | Detector confidence for this track |
| `video_status` | Review status of the video (`pending` / `processed` / `approved`) |

**Export summary** — one row per video × cluster, plus mission-wide totals as `video=ALL`:

| Column | Meaning |
|---|---|
| `mission_id`, `video`, `cluster_id`, `cluster_name` | Which mission/video/cluster this row is |
| `n_tracks` | Number of tracks in this cluster for this video (or mission-wide, for `ALL` rows) |
| `n_detections` | Total detections across those tracks |
| `max_n` | **MaxN** — the peak number of individuals from this cluster seen simultaneously in a single frame. This is the standard abundance metric used in baited/remote underwater video (BRUV) surveys. |

## Troubleshooting

- **`docker: command not found` / `Cannot connect to the Docker daemon`** — Docker Desktop isn't running. Launch it (Start menu on Windows, Applications on macOS — or `open -a Docker` on macOS) and wait ~30–60s for the whale icon to stop animating, then retry.
- **`docker compose build` is very slow the first time** — that's normal; it's downloading the base image and machine-learning libraries. Later builds are much faster.
- **Port is already in use / app won't start** — another program is using the port. On Docker the app uses **5001**; without Docker it uses **5000** (which macOS AirPlay Receiver also uses — see the no-Docker note above).
- **Windows: `curl` behaves strangely** — in PowerShell, `curl` is an alias for a different command. Use `curl.exe` for the Dropbox token step.
- **Code changes don't take effect** — with Docker, re-run `docker compose build` before `docker compose up`. Running directly (`python application/app.py`), the server doesn't auto-reload — stop it (`Ctrl+C`) and start it again.
- **A deployment/mission isn't showing up in the dropdowns** — confirm its folder has a `deployment.json` (for deployments) or an `*_embeddings.json` file (for missions to appear in Cluster Review); then restart the app so it picks up the new folder.
- **A video won't play in Cluster Review** — the app downloads and transcodes a low-res proxy from Dropbox the first time you open a video, which can take a few seconds. If it stays stuck on "generating" or errors out, double-check the deployment's Dropbox root and credentials — that download happens in the background and fails silently if the path is wrong.

---

## 📁 Editing code & key reminders

### First-time editor setup
1. Create and activate a virtual environment:
   ```
   python -m venv venv
   ```
   | | Activate | Deactivate |
   |---|---|---|
   | **Windows (PowerShell)** | `venv\Scripts\Activate.ps1` | `deactivate` |
   | **macOS** | `source venv/bin/activate` | `deactivate` |
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Making changes
1. Pull the latest: `git pull origin main`
2. Create a branch: `git checkout -b <branch-name>` (switch branches with `git checkout <branch-name>`)
3. Activate your virtual environment (if not already active — see above).
4. Make your edits.
5. Stage and commit: `git commit -a -m "<message>"`
6. Push: `git push`
7. On GitHub, open the **Pull requests** tab → **New pull request**, select your branch, describe your changes, and request reviews from your teammates.
