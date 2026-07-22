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
4. Build the Docker image **(first time only, or after code changes)**:
    ```
    docker compose build
    ```

### Running the app
```
docker compose up
```
Then open `http://localhost:5001` in your browser.

To stop: `docker compose down`

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

- **`Cannot connect to the Docker daemon...`** — Docker Desktop isn't running. Launch it from Applications (or `open -a Docker` in a terminal) and wait ~30–60s for the whale icon in the menu bar to stop animating, then retry.
- **Code changes don't seem to take effect** — if you're running the app directly (`python application/app.py`, not Docker), the server doesn't auto-reload. Stop it (`Ctrl+C`) and start it again. If you're using Docker, re-run `docker compose build` before `docker compose up`.
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
