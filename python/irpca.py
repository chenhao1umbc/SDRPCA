"""This is the python version of lrr using pytorch GPU because MATLAB 2018s always shows some funny errors"""
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
data_sets = ['EYB', 'coil', 'AR']

optdata={}
optdata['dataset'] = data_sets[0]
optdata['ind_dataset'] = 1
optdata['cv_fold'] = 5
optdata['max_iter'] = 1
optdata['rng'] = 0
optdata['o_per'] = 0.1
optdata['use_gpu'] = torch.cuda.is_available()
optdata['tol'] = 5e-6  # tolerance
optdata['rho'] = 1.1  # mu exponentially increase
optdata['max_mu'] = 1e8


# no outlier added just loading the data, in numpy format
data, labels = load_data()
knn = KNN(5)

o_per_sets = np.arange(0.0, 0.6, 0.1)
lam_sets = [2**i for i in range(3, -16, -1)]
cv_fold_sets = range(optdata['cv_fold'])
acc_all = []

for i in data_sets:
    optdata['dataset'] = i
    for o in o_per_sets:
        optdata['o_per'] = o
        for l in lam_sets:
            acc = 0
            for c in cv_fold_sets:
                # split the data, normalized and corrupted, in torch.tensor format
                xtr, y_tr, xte, y_te = data_gen(data, labels, optdata)
                Z, E = train_lrr(xtr, xtr, l, optdata)  # in the training it may use cuda
                P = get_prj(xtr@Z)
                knn.fit((P@xtr).t(), y_tr)
                y_hat = knn.predict((P@xte).t())
                acc = acc + metrics.accuracy_score(y_te, y_hat)
                torch.cuda.empty_cache()
            acc_all.append(acc/5)
            with open('lrr', 'a') as f:
                f.write('dataset is '+str(i)+'outlier percentage is '+str(o)+'lambda is '+str(l)+'current acc is '+str(acc/5)+'\n')

np.save('acc_irpca', acc_all)
