import os
import pandas
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from datetime import datetime
import pickle

FTRAIN = '~/kaggle/data/training.csv'
FTEST = '~/kaggle/data/test.csv'
FLOOKUP='~/kaggle/data/IdLookupTable.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y



def train():
        net1 = NeuralNet (
                layers = [
                        ('input',layers.InputLayer),
                        ('hidden', layers.DenseLayer),
                        ('output', layers.DenseLayer),
                        ],
        input_shape = (None, 9216),
        hidden_num_units = 100,
        output_nonlinearity = None,
        output_num_units =  30,
        update = nesterov_momentum,
        update_learning_rate = 0.1,
        update_momentum = 0.9,
        regression = True,
        max_epochs = 50,
        verbose = 1,
        )

        X, y = load()
        model = net1.fit(X,y)
        with open('model_pickle','wb') as file:
                pickle.dump(model,file,-1)


def predict():
    import pdb
    import pickle
    import itertools
    with open('model_pickle','rb') as file:
    	model=pickle.load(file)
    X,y = load()
   
    y_pred1 = model.predict(X)
    y_pred = np.hstack(y_pred1)
    columns =  [ "left_eye_center",         "right_eye_center",
		 "left_eye_inner_corner",   "left_eye_outer_corner"  ,
 		 "right_eye_inner_corner",  "right_eye_outer_corner" ,
 		 "left_eyebrow_inner_end",  "left_eyebrow_outer_end" ,
		 "right_eyebrow_inner_end", "right_eyebrow_outer_end",
		 "nose_tip",                "mouth_left_corner"    ,  
		 "mouth_right_corner",      "mouth_center_top_lip",   
		 "mouth_center_bottom_lip"
		]

    cols=[[col_name+'_x',col_name+'_y'] for col_name in columns]
    col_names = list(itertools.chain(*cols))        

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = pandas.DataFrame(y_pred2,col_names*2140)
     
    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
	    df[0][row.FeatureName][row.ImageId - 1],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = pandas.DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))



predict()
