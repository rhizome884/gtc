# Guitar Tuning Classification 

## Synthetic Audio
An audio example from the dataset, which was generated by sending guitar tablature data to a sample-based guitar plugin, is available [here](https://drive.google.com/file/d/1G_yRHhJQj9c0JJdJx_iZt3HFvwy_TcMn/view?usp=drive_link).

## Model Evaluation

### Install Requirements
You will need at least Python 3.8. We recommend installing Python and the required packages in a virtual environment using *venv* or *conda*, and *pip*.   	

Install the following packages after setting up your environment:
```bash
pip install tensorflow matplotlib scikit-learn seaborn
```

Clone the repo and change directory:
```bash
git clone https://github.com/rhizome884/gtc.git 
cd gtc
```

### Download Models and Datasets 
The test sets and trained models used for evaluation of the 5-class chromagram system can be downloaded from [here](https://drive.google.com/drive/folders/1bs8kPQcPk3Mr6a4m1QlQVrEXbJ5ro7Mc?usp=drive_link). After the *test-sets* and *models* folders have been downloaded and unzipped, put both folders inside the *gtc* directory. 

### Evaluate Model
Run model evaluation on synthetic data in your local environment:
```bash
python eval_synthetic_songs.py
``` 
 
Run model evaluation on authentic data in your local environment:
```bash
python eval_authentic_songs.py
```
