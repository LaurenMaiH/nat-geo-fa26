# [DSS x Pristine Seas] Benthic Species Clusterer
Species classification for deep sea footage. 

## 🛠️ Setup Instructions

#### Clone the repository:
1. Open your laptop's terminal
2. `cd` into the file directory you want to save this repository in
3. Clone the repository:
    ```
    git clone <SSH url>
    cd <repo-name>
    ```

> Already have the repo from before? Skip cloning and just get the latest changes instead:
> ```
> git pull origin main
> ```

## To run (Docker — recommended)

No Python or ffmpeg setup required. Just Docker Desktop.

### First time setup
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Create a `.env` file in the `nat-geo/` folder using `.env.example` as a template and fill in the Dropbox credentials:
    ```
    DROPBOX_APP_KEY=your_app_key_here
    DROPBOX_APP_SECRET=your_app_secret_here
    DROPBOX_REFRESH_TOKEN=your_refresh_token_here
    ```
    If you don't have these yet, here's how to generate them for your own Dropbox account:
    1. Go to the [Dropbox App Console](https://www.dropbox.com/developers/apps) and click **Create app**.
    2. Choose **Scoped access**, then **Full Dropbox** access (not "App folder" — the app needs to read mission video folders wherever they live in your Dropbox, not just inside a dedicated app folder). Name the app and create it.
    3. On the app's **Permissions** tab, enable `files.metadata.read` and `files.content.read`, then click **Submit**. The app only ever reads videos — no write scopes are needed.
    4. On the app's **Settings** tab, copy the **App key** and **App secret** into `.env` as `DROPBOX_APP_KEY` / `DROPBOX_APP_SECRET`.
    5. Get the refresh token (a one-time setup per app): visit this URL in your browser, swapping in your App key —
        ```
        https://www.dropbox.com/oauth2/authorize?client_id=YOUR_APP_KEY&response_type=code&token_access_type=offline
        ```
        Click **Allow**, and copy the authorization code it shows you.
    6. Exchange that code for a refresh token:
        ```
        curl https://api.dropbox.com/oauth2/token \
          -d code=THE_CODE_FROM_STEP_5 \
          -d grant_type=authorization_code \
          -d client_id=YOUR_APP_KEY \
          -d client_secret=YOUR_APP_SECRET
        ```
        The JSON response includes a `refresh_token` field — that's your `DROPBOX_REFRESH_TOKEN`. It doesn't expire, so this is a one-time step.
3. Download the sample deployment data:
    - Download the `DOEX0096_Palau` folder from [this Google Drive link](https://drive.google.com/drive/folders/1BD5iZZIn2RrYc0OxrK8cHqs3ObZ5Eu95?usp=sharing) (contains `deployment.json` and the `PLW_dscm_11` mission — crops, proxy videos, detections CSV, embeddings, and state.json)
    - Inside `nat-geo/`, create a folder named `processed_missions` if it doesn't already exist
    - Place the downloaded `DOEX0096_Palau` folder inside it, so the paths `nat-geo/processed_missions/DOEX0096_Palau/deployment.json` and `nat-geo/processed_missions/DOEX0096_Palau/PLW_dscm_11/` both exist

    This gives the app a deployment and mission to open by default. Missions are grouped under a "deployment" folder — a deployment is one Dropbox root you're pulling videos from (e.g. one client's account, or one expedition). To add more missions to this same deployment, download or process them the same way — each one gets its own folder under `processed_missions/DOEX0096_Palau/`. See "Connecting a different deployment" below for adding a separate one.
4. **(Windows — strongly recommended)** Choose where large files go. In `.env`, add these three lines, pointing at a drive with lots of free space (see [Disk space (Windows)](#disk-space-windows) for why):
    ```
    SCRATCH_DIR=D:/PristineSeas/scratch
    VIDEO_DIR=D:/PristineSeas/videos
    MODEL_DIR=D:/PristineSeas/model_cache
    ```
5. Build the Docker image **(first time, and again after every code change or `git pull`)**:
    ```
    docker compose build
    ```

### Running the app
```
docker compose up
```
Then open `http://localhost:5001` in your browser.

To stop: `docker compose down`

> **Docker does not pick up code changes on its own.** The code is copied into the image when it's built, so after editing code or running `git pull`, rebuild before starting: `docker compose up --build`. Only the code layers are rebuilt (seconds); the Python packages are reinstalled only if `requirements.txt` changes.

### Where files are stored

Everything large lives in ordinary folders on your computer, not inside Docker:

| Folder | Set with | Default | What's in it |
|---|---|---|---|
| Results | — | `processed_missions/` | Crops, detections CSV, embeddings, review state, video proxies |
| Scratch | `SCRATCH_DIR` | `scratch/` | Videos being downloaded and re-encoded. Emptied automatically after each video. |
| Videos | `VIDEO_DIR` | `videos/` | Local copies of mission videos (read-only to the app) |
| Models | `MODEL_DIR` | `model_cache/` | ~6 GB of model weights, downloaded on the first pipeline run and reused after that |

On Windows, write paths in `.env` with forward slashes (`D:/PristineSeas/scratch`), without quotes. Folders that don't exist yet are created automatically.

### Optional: use an NVIDIA GPU (Windows)

By default everything runs on the CPU. If the computer has an NVIDIA graphics card, the pipeline can use it and run much faster:

1. Install the latest NVIDIA driver for Windows from [nvidia.com/drivers](https://www.nvidia.com/drivers). Nothing else needs to be installed; Docker Desktop (WSL 2) passes the GPU through automatically.
2. Start the app with the GPU file added (the first build downloads a GPU version of PyTorch, which is several GB larger):
    ```
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
    ```
3. Run the pipeline and check the log: it should say `Using device: cuda`. If it says `cpu`, the GPU wasn't found. Update the driver and restart Docker Desktop.

The GPU build is for RTX 20-series cards and newer. For older cards (e.g. GTX 1080), add `TORCH_CUDA_VARIANT=cu126` to `.env` first. **Macs:** Docker on macOS cannot use the GPU at all, so use the normal `docker compose up`.

Switching between GPU and CPU mode rebuilds the image. Afterwards, run `docker image prune` to delete the old copy.

## Disk space (Windows)

Docker Desktop stores everything it keeps internally in a single file on `C:\`:

```
C:\Users\<your-username>\AppData\Local\Docker\wsl\disk\docker_data.vhdx
```

That file **grows but never shrinks on its own**, even after Docker deletes things. The steps below keep it small and get back space it has already used.

### 1. Send large files to D:\

Do step 4 of [First time setup](#first-time-setup): set `SCRATCH_DIR`, `VIDEO_DIR` and `MODEL_DIR` in `.env` to folders on `D:\`. Then the video downloads, temporary re-encodes and model weights are written to `D:\` instead of into Docker's file. Rebuild and restart afterwards (`docker compose up --build`).

### 2. (Optional) Move Docker's own file to D:\

In Docker Desktop: **Settings (gear icon) → Resources → Advanced → Disk image location**. Choose a folder on `D:\` (e.g. `D:\DockerData`), then click **Apply & restart**. Docker moves the file there, which can take a few minutes. The app's Docker image (a few GB) and Docker's other internal data will then live on `D:\`.

### 3. Get back space that's already used

Do this once now to recover the space from earlier runs, and again whenever `C:\` gets tight.

1. Open **PowerShell** and delete Docker's unused images, stopped containers and build cache:
    ```
    docker system prune -a
    ```
    Type `y` to confirm. This does **not** touch your results, videos or model weights, because those are in normal folders. The next `docker compose up --build` rebuilds the image (about 5–10 minutes).
2. Quit Docker Desktop: right-click the whale icon in the taskbar (near the clock), then click **Quit Docker Desktop**.
3. In PowerShell, run:
    ```
    wsl --shutdown
    ```
4. Shrink the file. This needs an **administrator** PowerShell: click Start, type `PowerShell`, right-click it, then click **Run as administrator**. Replace `<your-username>` with your Windows user name in the commands below. If you moved the disk image in step 2, use that folder instead.

    **Windows Pro / Enterprise / Education:**
    ```
    Optimize-VHD -Path "C:\Users\<your-username>\AppData\Local\Docker\wsl\disk\docker_data.vhdx" -Mode Full
    ```
    **Windows Home**, or if `Optimize-VHD` says it isn't recognized, run these lines one at a time:
    ```
    diskpart
    select vdisk file="C:\Users\<your-username>\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
    attach vdisk readonly
    compact vdisk
    detach vdisk
    exit
    ```
    Shrinking a large file can take 10+ minutes. When it finishes, check the file's size in File Explorer, then start Docker Desktop again.

    *Can't find the file?* Older Docker Desktop versions call it `C:\Users\<your-username>\AppData\Local\Docker\wsl\data\ext4.vhdx`.

### 4. Check how much space Docker is using

```
docker system df
```
This lists how much space Docker's images, containers and build cache take, and how much of each can be reclaimed. Add `-v` for a per-item breakdown.

### 5. Clean up after rebuilding

Every `docker compose build` (or `up --build`) leaves the previous version of the image behind. Every so often, run:
```
docker image prune
docker builder prune
```
The first command deletes old, untagged images; the second deletes Docker's cache of build steps. Neither touches your data. If space is still tight after that, repeat step 3.

## Connecting a different deployment

A "deployment" is a Dropbox root folder you're pulling mission videos from — e.g. a different client, or a different expedition. Each deployment keeps its own missions, crops, and clustering results completely separate from every other deployment.

- **Switching between deployments already set up**: use the "Deployment" dropdown in the app's top bar. Everything below it — missions, review, export — scopes to whichever deployment is selected.
- **Adding a brand-new deployment**: go to the Pipeline tab, and in the Deployment dropdown pick "+ New deployment…". Give it a name and the Dropbox root path for that deployment (e.g. `/SomeClientFolder`), fill in Mission ID as usual, and start the pipeline. The app creates the deployment's folder automatically the first time it runs — no manual setup needed.

## Exporting results

In the Cluster Review tab, once a mission is initialized, two buttons in the header download the clustering results as CSV:

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

- **`Cannot connect to the Docker daemon...`** — Docker Desktop isn't running. Launch it from Applications on a Mac (or `open -a Docker` in a terminal) or from the Start menu on Windows, and wait ~30–60s for the whale icon to stop animating, then retry.
- **Code changes don't seem to take effect** — neither setup reloads code automatically. With Docker, the code is copied into the image at build time, so the running container keeps the old code until you rebuild: `docker compose up --build`. If you're running the app directly (`python application/app.py`), stop it (`Ctrl+C`) and start it again.
- **Docker stopped partway through a run / `C:\` is full** — see [Disk space (Windows)](#disk-space-windows).
- **A deployment/mission isn't showing up in the dropdowns** — confirm its folder has a `deployment.json` (for deployments) or an `*_embeddings.json` file (for missions to appear in Cluster Review); then restart the app so it picks up the new folder.
- **A video won't play in Cluster Review** — the app has to download and transcode a low-res proxy from Dropbox the first time you open a video, which can take a few seconds. If it stays stuck on "generating" or shows an error, double check the deployment's Dropbox root and credentials are correct — that download is happening in the background and failing silently if the path is wrong.


## 📁 Editing Code & Key Reminders

### First time editing
1. Create and activate a virtual environment
    
    Create a new virtual environment: `python3 -m venv venv`

    Activate: `source venv/bin/activate`

    To deactivate later: `deactivate`
2. Install dependencies:

    ```pip install -r requirements.txt```

### Editing Code
1. Pull the latest changes

    ```git pull origin main```

2. Create a new branch:

    Create a new branch: ```git checkout -b <branch-name>```
    
    Switch between branches: ```git checkout <branch-name>```
3. Activate your virtual environment (if not activated already)

    Activate: `source venv/bin/activate`

4. Make your edits
5. Commit and add your new code:

    ```git commit -a -m <name>```

6. Push your new edits

    ```git push```

7. Open Github and go to `pull requests` tab. 
8. Click "New Pull Request", select your branch, and add a description of the changes.
9. Request reviews/approvals from your teammates.
