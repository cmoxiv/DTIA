
from tqdm import tqdm
import time

files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]

files = [f"file{i}.txt" for i in range(1000)]
with tqdm(total=len(files)) as pbar:
    for file in files:
        pbar.set_description(f"Processing {file}")
        time.sleep(0.01)  # simulate file processing
        pbar.update(1)
