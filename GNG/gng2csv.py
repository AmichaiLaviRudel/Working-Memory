# Purpose: READ GNG DATA AND CONVERT TO DataFrame

# Run before the load_gng_data function in matlab
import scipy.io as spio
import pandas as pd
import os
import matlab.engine

file_path = r"Z:\Shared\Amichai\Benne project\Naive\behav\juv_1\juv_1\230625\novice_disc_rec_conc.mat"

eng = matlab.engine.start_matlab()
eng.load_gng_data(file_path, nargout=0)

# Change the file_path ending to end with _formmated.mat
path, name = os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]
formatted_file_path = os.path.join(path, f"{name}_formmated.mat")

data = spio.loadmat(formatted_file_path, squeeze_me=True)
# run over data and force to be length 1
for key in data:
  data[key] = [data[key]]

# remove '__header__', '__version__', '__globals__'
data.pop('__header__')
data.pop('__version__')
data.pop('__globals__')

# dict to dataframe
data = pd.DataFrame.from_dict(data)

path, name = os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]
data.to_csv(f"{os.path.join(path, name)}_for_db.csv", index=False)
os.remove(formatted_file_path)

