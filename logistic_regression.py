import numpy as np
class Logistical_Regression:

    def __init__(self,obj,itr=10000,aphla=0.1):
        obj.setting_data()
        print("it is merged matrix of two matrix X1 and X2 ", obj.merged_matrix.shape)
        print("It is the shape of label which is take as Y", obj.training_lab.shape)
        print("It is the shape of theta which is take as theta", obj.theta.shape)
        self.theta =Logistical_Regression.finding_theta(itr ,aphla,obj.merged_matrix,obj.theta,obj.training_lab)
        theta=self.theta


    def sigmod(hip_cur):
        return 1 / (1 + np.exp(-hip_cur))

    def finding_theta(itr ,aphla,merged_matrix,theta,training_lab):
        for i in range(itr):
            appro_y = Logistical_Regression.sigmod((np.dot(merged_matrix, theta)))
            gradient = np.dot(merged_matrix.T, (appro_y - training_lab.T))
            theta = theta - (aphla/ merged_matrix.shape[0]) * gradient
        return theta