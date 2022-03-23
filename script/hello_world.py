import sys
from time import process_time

sys.path.insert(0, '/home/aboulin/')

theta = 2
n_sample = 10000

bv_copula = Frank(theta = theta, n_sample = n_sample)

from CoPY.copy.multivariate.mv_archimedean import Frank

d = 2

mv_copula = Frank(theta = theta, n_sample = n_sample, d = d)

list = {'bv_copula' : bv_copula, 'mv_copula' : mv_copula}

def compare_time(list) :

    for name, object in list.items() :
        print('computing time for the {}'.format(name))
        t = process_time()
        sample = object.sample_unimargin()
        elapsed_time = process_time() - t
        print('the elapsed time is equal to {}'.format(elapsed_time))

compare_time(list)