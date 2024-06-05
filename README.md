# Installation Instructions

Ensure you have python downloaded: https://www.python.org/downloads/

Before you can run, you must install the following librairies:

### pandas
python -m pip install pandas
### sci-kit learn
python -m pip install sci-kit learn

# Running Instructions

Run the following command to see the output for each model:

### Linear
python linear-regression.py

### Decision Tree
python decision-tree-regression.py

### Random Forest
python random-forest-regression.py

You should see the participants characteristics, along with the model's predicted value. 

## Changing between squat and deadlift

The program is defaulted to print out the participants squat predictions. To print out the participants deadlift prediction, 
change the predict_variable to 'BestDeadliftKg' (located in line 8)

#### predict_variable = 'BestSquatKg'  # BestSquatKg or BestDeadliftKg