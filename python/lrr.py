"""This is the python version of lrr using pytorch GPU because MATLAB 2018s always shows some funny errors"""
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_sets = ['EYB', 'coil', 'AR']

optdata={}
optdata['data_set'] = data_sets[0]
optdata['ind_dataset'] = 1
optdata['cv_fold'] = 5
optdata['max_iter'] = 1000
optdata['rng'] = 0
optdata['o_per'] = 0.1
optdata['use_gpu'] = torch.cuda.is_available()
optdata['tol'] = 5e-6  # tolerance
optdata['rho'] = 1.1  # mu exponentially increase
optdata['max_mu'] = 1e8


# no outlier added just loading the data, in numpy format
data0, labels0 = load_data()

# split the data, normalized and corrupted, in torch.tensor format
xtr, xtr_labels, xte, xte_labels = data_gen(data0, labels0, optdata)




knn = KNN(5)
knn.fit(p@xtr,xtr_labels)
y_hat = knn.predict(xte)
acc = metrics.accuracy_score(y, y_hat)