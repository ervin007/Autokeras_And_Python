from tensorflow.keras.models import load_model

Custom_Objects = ak.CUSTOM_OBJECTS

# If you have custom metrics add each metric accordingly
# Custom_Objects["f1_score"] = f1_score

loaded_model = load_model("Models/First_Run", custom_objects=Custom_Objects)

print(loaded_model.evaluate(test_file_path, "survived"))