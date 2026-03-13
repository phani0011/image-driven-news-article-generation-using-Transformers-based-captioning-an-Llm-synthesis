import os

TOTAL_CHUNKS = 20   # 30000 / 1500

for chunk in range(TOTAL_CHUNKS):
    print(f"\n===== Starting chunk {chunk} =====\n")
    result = os.system(f"python train.py --chunk {chunk}")

    if result != 0:
        print("Training stopped due to error.")
        break

print("\nTraining process finished.")

