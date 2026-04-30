# nat-geo
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

## To run: 

1. create env
python -m venv venv
source venv/bin/activate

2. install deps
pip install -r requirements.txt

3.  install ffmpeg (mac)
brew install ffmpeg

4. install default and finetuned model and place them into the seaanimals_proto folder
    
    Default Yolo-World model: https://drive.google.com/file/d/1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406/view?usp=drive_link

    Finetuned Yolo-World model (trained on JAMSTEC dataset on deep-sea animals): https://drive.google.com/file/d/1rCyq4GZcG5UCqrl2SNNjmaHkLOZbWgyv/view?usp=drive_link

4. cd into seaanimals_proto

5. run app
streamlit run model_runner.py

#### IF RUNNING INTO SSL ERROR ####
1. 
``` 
python3 -m pip install --upgrade certifi requests urllib3 /Applications/Python\ 3.12/install\ Certificates.command 
```

2. 
``` 
mkdir -p ~/.cache/clipcurl -L --fail -o ~/.cache/clip/ViT-B-32.pt "https://openaipublic.azureedge.net/clip/models/40d365715913c9-da98579312b702a82c18be219c-c2a73407c4526f58eba950af/ViT-B-32.pt" 
```

3. 
``` 
ls -lh ~/.cache/clip/ViT-B-32.ptpython3 -c "import clip; print('before'); model, preprocess = clip.load ('ViT-B/32'); print('after')"
```


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
