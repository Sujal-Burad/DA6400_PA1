## Getting Started

Before running the project, ensure you have all necessary dependencies installed by executing the following command in your terminal:

pip install -r requirements.txt



## Running SARSA and Q-Learning

To run SARSA and Q-learning for each environment:

1. Navigate into the corresponding environment directory.
2. Execute the `main.py` file using Python with command arguments:


## Hyperparameter Tuning

For hyperparameter tuning of the algorithms:

1. Open the `hyperparameter_tuning_sarsa.py` or `hyperparameter_tuning_q_learning.py` file.
2. Modify the parameters you wish to fine-tune.
3. Adjust the sweep method as required. (We used the 'bayes' sweep method as it is typically more efficient than grid or random search. It uses past results to inform future searches, focusing on the most promising areas of the hyperparameter space.
