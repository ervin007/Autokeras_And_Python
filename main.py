import tensorflow as tf
import autokeras as ak
import kerastuner

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

tensorboard_callback_train = tf.keras.callbacks.TensorBoard(log_dir='Tensorboard//logs_autokeras_f1_score_10%')
tensorboard_callback_test = tf.keras.callbacks.TensorBoard(log_dir='Tensorboard//logs_autokeras_f1_score_10%')

Early_Stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=101)

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    project_name = "Projects/First_Run",
    objective=kerastuner.Objective('accuracy', direction='max'),
    metrics=["accuracy"],
    overwrite=True,
    max_trials=1001)

clf.fit(
    train_file_path # or  train_x and train_y
    epochs=151,
    verbose=2,
    callbacks=[tensorboard_callback_train, Early_Stopping],
    batch_size=1001
    )


clf_best_model = clf.export_model()
clf_best_model.save("Models/First_Run", save_format="tf")
print(accuracy=clf.evaluate(test_file_path, "survived"))