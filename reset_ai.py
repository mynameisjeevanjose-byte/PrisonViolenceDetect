import os

print("Resetting AI Data...")
files_to_delete = ['X_data.npy', 'y_data.npy', 'brawl_lstm_model.h5']

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f" -> Deleted: {file}")
    else:
        print(f" -> {file} already missing/deleted.")

print("\nReset Complete! You can now start fresh with collect_data.py.")