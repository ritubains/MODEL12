import numpy as np
class Testing:

    def __init__(self,theta,obj):
        result = self.sigmod((np.dot(obj.merged_mat_test, theta)))
        self.res = np.round(result)
        self.give_pert(obj)

    def sigmod(self,hip_cur):
        return 1 / (1 + np.exp(-hip_cur))

    def give_pert(self,obj):
        count = 0
        for i in range(self.res.shape[0]):
            if (int(self.res[i]) == int((obj.testing_lab.T)[i])):
                count = count + 1
        print("answer" ,(count / self.res.shape[0]) * 100)