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
3. Place the `processed_missions/` data folder inside `nat-geo/`
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
