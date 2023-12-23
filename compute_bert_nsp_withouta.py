"""
Take the predictions from BERT NSP and remove all the 'A' predictions, renormalise.

Due to some unknown bug this setup was not working when trained from scratch, so let's try to recreate it this way.
"""
import pandas as pd
import numpy as np

predictions_to_load = "./output_predictions/distantsupervision_both_rerun_dev_BERT NSP"
predictions_to_save = "./output_predictions/distantsupervision_both_withouta_rerun_dev_BERT NSP"

data = pd.read_csv(predictions_to_load, index_col=0)
print(data)
data = data.drop(["gold_for_class_0", "predictions_for_class_0"], axis=1)
data.to_csv(predictions_to_save)
