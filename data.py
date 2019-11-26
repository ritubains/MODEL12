import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self,path):
        self.path=path

    def read_data(self):
        self.mydata=pd.read_csv(self.path,index_col=False)
        self.set_data()

    def set_data(self):
        training_parameter,testing_parameter,training_labels,testing_labels=train_test_split(self.mydata[["X","Y"]],self.mydata[["label"]],test_size=0.2)

        X_value=np.matrix(training_parameter["X"])
        Y_value=np.matrix(training_parameter["Y"])

        self.merged_matrix=np.concatenate((np.matrix(np.ones(training_parameter.shape[0])).T,X_value.T,Y_value.T),axis=1)
        self.label=np.matrix(training_labels["label"])
        self.beta=np.zeros(self.merged_matrix.shape[1])
        self.beta=self.beta.reshape(self.merged_matrix.shape[1],1)

        X_test_value=np.matrix(testing_parameter["X"])

        Y_test_value=np.matrix(testing_parameter["Y"])


        self.merged_matrix_test=np.concatenate((np.matrix(np.ones(X_test_value.shape[1])).T,X_test_value.T,Y_test_value.T),axis=1)

        print(self.merged_matrix.shape)
        print("this is file")









