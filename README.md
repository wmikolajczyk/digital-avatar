# digital-avatar
Create a Digital Avatar by fine-tuning the Transformer language model on conversations data

Steps:

1. Create `data` dir and move there all .zip files with Facebook data

2. `git clone git@github.com:MasterScrat/Chatistics.git`

3. `pip install -r requirements.txt`

4. ```bash
   python prepare_data_for_chatistics.py
   cd Chatistics
   python parse.py messenger --max-exported-messages=1000000000
   python export.py -f json
   cd ..
   python preprocess_chatistics_export.py
   ```
