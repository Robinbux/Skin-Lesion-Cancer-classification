import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.data_generator import ColorSpace, SkinLesionDataSequence, DatasetType

def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

batch_size = 32
epochs = 10

for color_space in ColorSpace:
    print(f"Training model for {color_space.value} color space...")
    
    # Adjust input_shape as needed
    input_shape = (224, 224, 3)  
    model = build_model(input_shape=input_shape)
    
    checkpoint_filepath = f'best_model_{color_space.value}.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_accuracy', mode='max')
    
    train_sequence = SkinLesionDataSequence(base_dir='data', dataset_type=DatasetType.Train, color_space=color_space, batch_size=batch_size, normalize=True)
    val_sequence = SkinLesionDataSequence(base_dir='data', dataset_type=DatasetType.Validation, color_space=color_space, batch_size=batch_size, normalize=True)
    
    # Train the model
    model.fit(train_sequence, epochs=epochs, validation_data=val_sequence, callbacks=[model_checkpoint_callback])
    
    print(f"Best model for {color_space.value} color space saved as {checkpoint_filepath}")

models = {}
for color_space in ColorSpace:
    model_path = f'best_model_{color_space.value}.keras'
    models[color_space.value] = load_model(model_path)

# Initialize an empty DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
metrics_list = []

threshold = 0.5

# Gather predictions from each model
individual_preds = {}
for color_space, model in models.items():
    test_sequence = SkinLesionDataSequence(base_dir='data', dataset_type=DatasetType.Test, color_space=ColorSpace(color_space), batch_size=batch_size, normalize=True)
    true_labels = test_sequence.labels
    preds = model.predict(test_sequence)
    individual_preds[color_space] = (preds > threshold).astype(int)

    acc = accuracy_score(true_labels, individual_preds[color_space].flatten())
    prec = precision_score(true_labels, individual_preds[color_space].flatten())
    rec = recall_score(true_labels, individual_preds[color_space].flatten())
    f1 = f1_score(true_labels, individual_preds[color_space].flatten())

    metrics_list.append({'Model': color_space, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# Simple average ensemble
ensemble_preds = np.mean(list(individual_preds.values()), axis=0)
final_ensemble_preds = np.where(ensemble_preds > threshold, 1, 0).flatten()

ensemble_acc = accuracy_score(true_labels, final_ensemble_preds)
ensemble_prec = precision_score(true_labels, final_ensemble_preds)
ensemble_rec = recall_score(true_labels, final_ensemble_preds)
ensemble_f1 = f1_score(true_labels, final_ensemble_preds)

metrics_list.append({'Model': 'Ensemble', 'Accuracy': ensemble_acc, 'Precision': ensemble_prec, 'Recall': ensemble_rec, 'F1': ensemble_f1})
metrics_df = pd.DataFrame(metrics_list)

# Bayesian Model Averaging Ensemble
weights = [metrics['Accuracy'] for metrics in metrics_list[:-1]]  # Exclude Ensemble from metrics_list
total_weight = sum(weights)
normalized_weights = [w / total_weight for w in weights]

prepared_preds = [pred.squeeze() for pred in individual_preds.values()]

bma_preds = np.average(prepared_preds, axis=0, weights=normalized_weights)
final_bma_preds = np.where(bma_preds > threshold, 1, 0).flatten()

bma_metrics = {
    'Model': 'Bayesian Model Averaging',
    'Accuracy': accuracy_score(true_labels, final_bma_preds),
    'Precision': precision_score(true_labels, final_bma_preds),
    'Recall': recall_score(true_labels, final_bma_preds),
    'F1': f1_score(true_labels, final_bma_preds)
}

bma_df = pd.DataFrame([bma_metrics])

metrics_df = pd.concat([metrics_df, bma_df], ignore_index=True)
metrics_df

# ## Results
# Same results for plain average ensemble and Bayesian Model Averaging.


