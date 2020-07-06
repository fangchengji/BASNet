import os
from glob import glob

pth_file = glob("./saved_models/basnet_bsi/*.pth")
for i in range(0, len(pth_file), 5):
    print("\n==============================")
    print(f"start evaluating {pth_file[i]}")
    os.system(f"python basnet_test.py --model_dir {pth_file[i]}")
    os.system(f"python basnet_evaluate.py")
