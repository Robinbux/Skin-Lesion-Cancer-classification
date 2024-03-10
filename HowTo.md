# How to
The final result can be seen in final_data_augmentation.ipynb. The notebook loads the previously trained models, evaluates their performance on the test set, and calculates various metrics (accuracy, precision, recall, and F1-score) for each individual model and ensemble approaches.

The other notebooks I left for baseline comparisons.

Before running the final notebook, you need to run the following scripts in this order:
1. preprocess.py
2. color_conversions.py
3. create_validation_set.py

After that, you can run the final_data_augmentation.ipynb notebook.
If you want you can comment out the training part, and directly load the pretrained models I provided via Google drive.

