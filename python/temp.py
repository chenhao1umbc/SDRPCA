"""This is the python version of lrr using pytorch GPU because MATLAB 2018s always shows some funny errors"""
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_sets = ['coil', 'AR','EYB', ]

optdata={}
optdata['dataset'] = data_sets[0]
optdata['ind_dataset'] = 1
optdata['cv_fold'] = 1
optdata['max_iter'] = 3
optdata['rng'] = 0
optdata['o_per'] = 0.1
optdata['use_gpu'] = torch.cuda.is_available()
optdata['tol'] = 5e-6  # tolerance
optdata['rho'] = 1.1  # mu exponentially increase
optdata['max_mu'] = 1e8

def train_temp(X, A, lamb, optdata):
    # initialization
    X = X.cuda() if optdata['use_gpu'] else X
    A = A.cuda() if optdata['use_gpu'] else A

    m, n = X.shape
    mu = 0.1 * lamb
    atx = A.t()@X
    inv_a = torch.inverse(A.t()@A + torch.eye(n)) if not optdata['use_gpu'] else torch.inverse(A.t()@A + torch.eye(n).cuda())

    J = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    Z = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    E = torch.zeros(m, n).cuda() if optdata['use_gpu'] else torch.zeros(m, n)
    Y1 = torch.zeros(m, n).cuda() if optdata['use_gpu'] else torch.zeros(m, n)
    Y2 = torch.zeros(n, n).cuda() if optdata['use_gpu'] else torch.zeros(n, n)
    diff = torch.zeros(optdata['max_iter'])

    # body of algorithm
    for i in range(optdata['max_iter']):
        t = time.time()
        # update J
        temp = Z + Y2 / mu
        U, sigma, V = np.linalg.svd(temp.cpu().numpy())
        if optdata['use_gpu']:
            U, sigma, V = torch.from_numpy(U).cuda(), torch.from_numpy(sigma).cuda(), torch.from_numpy(V).cuda()
        else:
            U, sigma, V = torch.from_numpy(U), torch.from_numpy(sigma), torch.from_numpy(V)

        svp = sigma[sigma>1/mu].shape[0]
        if svp > 1:
            sigma = sigma[:svp] - 1/mu
            J = U[:, :svp] @ sigma.diag() @ V[:, :svp].t()
        else:
            svp, sigma = 1, 0
            J = U[:, :svp] *sigma @ V[:, :svp].t()

        # update Z
        Z = inv_a @ (atx - A.t()@E+J+(A.t()@Y1-Y2) / mu)

        # update E
        xmaz = X - A @ Z
        temp = xmaz + Y1 / mu
        E = max0(temp - lamb / mu) + min0((temp + lamb / mu))

        # when to stop
        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(leq1.abs().max(), leq2.abs().max())
        diff[i] = stopC
        if i > 10 and (abs(diff[i] - diff[i - 10]) / abs(diff[i]) < 1e-3):
            break
        if (stopC < optdata['tol']).item():
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            mu = np.minimum(optdata['max_mu'], mu * optdata['rho'])
            if not i%30 : print('cost:', stopC, 'current iter', i, 'iteration time is ', time.time()-t)
    return Z.cpu(), E.cpu()

# no outlier added just loading the data, in numpy format
data, labels = load_data()
knn = KNN(5)

o_per_sets = np.arange(0.0, 0.6, 0.1)
lam_sets = [2**i for i in range(1, -1, -1)]
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
                Z, E = train_temp(xtr, xtr, l, optdata)  # in the training it may use cuda
                P = get_prj(xtr@Z)
                knn.fit((P@xtr).t(), y_tr)
                y_hat = knn.predict((P@xte).t())
                acc = acc + metrics.accuracy_score(y_te, y_hat)
                torch.cuda.empty_cache()
            acc_all.append(acc/5)
            # with open('temp_log', 'a') as f:
            #     f.write('dataset is '+str(i)+'outlier percentage is '+str(o)+'lambda is '+str(l)+'current acc is '+str(acc/5)+'\n')

np.save('acc_temp', acc_all)


