# Guitar Tuning Classification 

## Model Evaluation

### Install Requirements
You will need at least Python 3.8. We recommend installing Python and the required packages in a virtual environment using *venv* or *conda*, and *pip*.   	

Install the following packages after setting up your environment:
```bash
pip install tensorflow matplotlib scikit-learn seaborn
```

Clone the repo and change directory:
```bash
git clone https://github.com/rhizome884/dlfm24-reproducible.git
cd dlfm24-reproducible
```

### Download Models and Datasets 
The test sets and trained models used for evaluation of the 5-class chromagram system can be downloaded from [here](https://drive.google.com/drive/folders/1bs8kPQcPk3Mr6a4m1QlQVrEXbJ5ro7Mc?usp=drive_link). After the *test-sets* and *models* folders have been downloaded and unzipped, put both folders inside the *dlfm24-reproducible* directory. 

### Evaluate Model
Run model evaluation on synthetic data in your local environment:
```bash
python eval_synthetic_songs.py
``` 
 
Run model evaluation on authentic data in your local environment:
```bash
python eval_authentic_songs.py
```  
