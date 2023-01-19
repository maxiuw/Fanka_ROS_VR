import numpy as np
import random
import os, shutil
def make_scene():
    path = "/media/raghav/m2/VRHRI_Rebecca/Assets/Scripts/rebecca_tests"
    try: 
        os.mkdir(path + "/setup_folder")  
    except FileExistsError:
        shutil.rmtree(path + "/setup_folder")
        os.mkdir(path + "/setup_folder")  
    path += "/setup_folder"
    poses = [[0, 0.5], [-0.1, 0.55], [0.1, 0.6]]
    classes = ["banana", "CubeDetected", "Food_Apple_Red"]
    # start with the order in which they won't be detected
    random.shuffle(classes)
    for i in range(len(classes)):
        # shuffle poses 
        random.shuffle(poses)
        with open(f"{path}/setup_{i}.txt", "w") as file, open(f"{path}/missing_{i}.txt", "a") as missing:
            for j in range(len(classes)):
                # remove object i or rather not include it 
                if (j == i):
                    missing.write(f"{classes[j]}: {poses[j][0]}, {poses[j][1]}")               
                    continue
                # save the bb for each detected class 
                file.write(f"{classes[j]}: {poses[j][0]}, {poses[j][1]}\n")
        file.close()

if __name__ == "__main__":
    make_scene()
