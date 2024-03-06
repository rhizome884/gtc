import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
from statistics import mode

from dataloader import prepare_test_set
import tensorflow as tf

# Choose model: 1-5
model_num = input("Choose a model (1-5): ")
song_name = []
song_preds = []
song_targets = []
max_song_preds = []

X_test, y_test, clips = prepare_test_set('test-sets/test-authentic.json')
model_path = 'models/' + 'model_0' + model_num + '.h5'
model = tf.keras.models.load_model(model_path)
# evaluate model accuracy on test set
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

y_preds = model.predict(X_test)
# get the prediction that occurs most for each song and store in list
for i, pred in enumerate(y_preds):
    song_preds.append(np.argmax(pred))
    clip = clips[i].split('_')[0]
    # print(clip)
    if i+1 != len(y_preds):
        clip_next = clips[i+1].split('_')[0]
    else:
        most_common = mode(song_preds)
        max_song_preds.append(most_common)
        song_targets.append(y_test[i])
        break
    if clip != clip_next:
        most_common = mode(song_preds)
        max_song_preds.append(most_common)
        song_targets.append(y_test[i])
        song_preds = []

print("Number of songs:", len(max_song_preds))
print()
print('Predictions:', max_song_preds)
print('Targets:    ', song_targets)
print()
# convert list to np array
y_preds = np.array(max_song_preds)

# report metrics
report = classification_report(song_targets, max_song_preds, output_dict=True, zero_division=0.0)
final_report = pd.DataFrame(report).transpose()
print(final_report)
# save report as csv
# report_name = 'report_real_0' + model_num + '.csv'
# final_report.to_csv(report_name, sep='\t', encoding='utf-8')

# confusion matrix
confusion_matrix = metrics.confusion_matrix(song_targets, max_song_preds)
# cm_name = 'confusion-matrix_real_0' + model_num + '.png'
# print(confusion_matrix)
heatmap = sns.heatmap(confusion_matrix, cmap="Blues", cbar=False, annot=True, fmt='g')
plt.xticks(np.arange(5)+0.5, ["DADGAD", "Drop D", "Open D", "Open G", "Standard"])
plt.yticks(np.arange(5)+0.5, ["DADGAD", "Drop D", "Open D", "Open G", "Standard"])
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
# plt.savefig(cm_name)
plt.show()

