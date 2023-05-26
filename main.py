import pandas as pd
import math
import random
import numpy as np

#final project

#predicts whether tumor is malignant or benign given certain parameters
class CancerDiagnosis:
    # number of features in the model
    num_feats = 10

    # multiple of inputs to use for number of hidden nodes
    hidden_pct = 3.7

    # number of output nodes
    num_outputs = 1

    # number of training epochs
    num_epochs = 1000

    # rate at which weights are adjusted
    learn_rate = 1.0

    # inputs nodes
    inputs = []

    # weights of input nodes to hidden nodes
    weights_input = []

    # hidden nodes
    hidden = [0] * int(num_feats * hidden_pct)

    # weights of hidden nodes to output nodes
    weights_hidden = []

    # output nodes
    outputs = []

    # runs sigmoid function for forward propagation
    # @param x the input value
    # @return sigmoid(x)
    def sigmoid(self, x):
        return (1 / (1 + math.exp(x * -1)))

    # runs sigmoid derivative function for backward propagation
    # @param x the input value
    # @return sigmoid derivative of x
    def sigmoid_derivative(self, x):
        return (x * (1 - x))

    # reads data from input file
    # @param filename: the file to read
    # @return features, labels: a list of scaled features, and labels and id
    def read_data(self, filename):
        file = pd.read_csv(filename)
        file['diagnosis'] = [1 if diagnosis == 'M' else 0 for diagnosis in file['diagnosis']]
        labels = file.diagnosis.values.tolist()
        id = file.id.values.tolist()

        file.drop(
            ['id', 'diagnosis', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
             'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
             'texture_worst',
             'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
             'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'], inplace=True, axis=1)

        my_maxes = CancerDiagnosis.get_maxes(self, filename)

        column_names = file.columns.tolist()

        for i in range(len(my_maxes)):
            file[column_names[i]] = [(value / my_maxes[i]) for value in file[column_names[i]]]

        features = file.values.tolist()

        return features, labels, id

    #gets maximum value for parameters in dataset
    # param file: file to read
    # return: array of maximum values
    def get_maxes(self, filename):
        file = pd.read_csv(filename)
        my_maxes = file[
            ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
             'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']].max().tolist()

        return my_maxes

    # converts tumor data to a string
    # param label: diagnosis of tumor
    # param id: id of tumor
    # @return string containing all properties
    def to_string(self, label, id):
        string = 'ID: %s, ' % str(id)
        if label == 1:
            string += 'Malignant '
        else:
            string += 'Benign '

        return string

    #creates training and testing sets
    # param features: scaled feature set
    # param labels: diagnoses of tumors
    # param id: ids of tumors
    # param bool: True or False to return either training or testing set
    # return: if bool == True, train features and labels; if bool == False, test features, labels, and id
    def train_test_sets(self, features, labels, id, bool):
        train_features = features[:-10]
        train_labels = labels[:-10]
        test_features = features[-10:]
        test_labels = labels[-10:]
        test_id = id[-10:]

        if bool:
            return train_features, train_labels
        elif not bool:
            return test_features, test_labels, test_id

    #sets up network structure
    def setup_network(self):
        for i in range(int(CancerDiagnosis.num_feats)):
            input_array = [random.random() for num in range(int(CancerDiagnosis.num_feats * CancerDiagnosis.hidden_pct))]
            CancerDiagnosis.weights_input.append(input_array)

        for i in range(int(CancerDiagnosis.num_feats * CancerDiagnosis.hidden_pct)):
            input_array = [random.random() for num in range(CancerDiagnosis.num_outputs)]
            CancerDiagnosis.weights_hidden.append(input_array)

    # runs forward propagation algorithm for a given element
    # returns: the vector of output values
    def forward_propagation(self, feature_set):
        hidden_weighted_sum = np.matmul(feature_set, CancerDiagnosis.weights_input).tolist()
        for i in range(len(CancerDiagnosis.hidden)):
            CancerDiagnosis.hidden[i] = CancerDiagnosis.sigmoid(self, hidden_weighted_sum[i])
        output_weighted_sum = np.matmul(CancerDiagnosis.hidden, CancerDiagnosis.weights_hidden).tolist()
        CancerDiagnosis.outputs = [CancerDiagnosis.sigmoid(self, output_weighted_sum[0])]

    # runs back propagation algorithm to update weights
    # param label_set: the labels to use for cost calculation
    def back_propagation(self, label_set):
        output_error = CancerDiagnosis.sigmoid_derivative(self, CancerDiagnosis.outputs[0]) * (label_set - CancerDiagnosis.outputs[0])

        hidden_errors = []
        for hidldx in range(len(CancerDiagnosis.hidden)):
            arr = CancerDiagnosis.weights_hidden[hidldx][0] * output_error * CancerDiagnosis.sigmoid_derivative(self, CancerDiagnosis.hidden[hidldx])
            hidden_errors.append(arr)
            CancerDiagnosis.weights_hidden[hidldx][0] += (CancerDiagnosis.learn_rate * CancerDiagnosis.hidden[hidldx] * output_error)
            for inldx in range(len(CancerDiagnosis.inputs)):
                CancerDiagnosis.weights_input[inldx][hidldx] += (CancerDiagnosis.learn_rate * CancerDiagnosis.inputs[inldx] * hidden_errors[hidldx])

    #runs the training algorithm on a network
    #param filename: the input file of training data
    def train_neural_network(self, filename):
        CancerDiagnosis.setup_network(self)
        features, labels, id = CancerDiagnosis.read_data(self, filename)
        train_features, train_labels = CancerDiagnosis.train_test_sets(self, features, labels, id, True)
        for i in range(1, CancerDiagnosis.num_epochs):
            count = 0
            for idx in range(len(train_features)):
                CancerDiagnosis.forward_propagation(self, train_features[idx])
                CancerDiagnosis.back_propagation(self, train_labels[idx])
                if round(CancerDiagnosis.outputs[0]) == train_labels[idx]:
                    count += 1
            if (i != 0) and (i % 100 == 0):
                print(count/(len(train_features)))

    #runs the test algorithm on a network
    #param filename: the input file of testing data
    def test_neural_network(self, filename):
        features, labels, id = CancerDiagnosis.read_data(self, filename)
        features, labels, id = CancerDiagnosis.train_test_sets(self, features, labels, id, False)
        count = 0
        for idx in range(len(features)):
            CancerDiagnosis.forward_propagation(self, features[idx])
            output_string = CancerDiagnosis.to_string(self, labels[idx], id[idx])

            if labels[idx] == 1:
                actual = 'Malignant'
            else:
                actual = 'Benign'
            if round(CancerDiagnosis.outputs[0]) == 1:
                expected = 'Malignant '
            else:
                expected = 'Benign '

            output_string += 'Actual: ' + actual + ', Expected: ' + expected

            if round(CancerDiagnosis.outputs[0]) == labels[idx]:
                count += 1
                output_string += ' MATCH'

            print(output_string)
        print(count / (len(features)))

        bool = input('Do you want to input the parameters of a tumor? (Y/N)')
        if bool == 'Y':
            CancerDiagnosis.check_tumor(self, filename)

    #interactive program that allows user to input features of tumor
    #param filename: file to read
    def check_tumor(self, filename):
        radius_mean = float(input('Enter radius mean: '))
        texture_mean = float(input('Enter texture mean: '))
        perimeter_mean = float(input('Enter perimeter mean: '))
        area_mean = float(input('Enter area mean: '))
        smoothness_mean = float(input('Enter smoothness mean: '))
        compactness_mean = float(input('Enter compactness mean: '))
        concavity_mean = float(input('Enter concavity mean: '))
        concave_points_mean = float(input('Enter concave points mean : '))
        symmetry_mean = float(input('Enter symmetry mean: '))
        fractal_dimension_mean = float(input('Enter fractal dimension mean: '))

        features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
         concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]

        my_maxes = CancerDiagnosis.get_maxes(self, filename)
        for i in range(len(features)):
            features[i] /= my_maxes[i]

        CancerDiagnosis.forward_propagation(self, features)

        if round(CancerDiagnosis.outputs[0]) == 1:
            expected = 'Malignant '
        else:
            expected = 'Benign '

        print('Expected diagnosis: ' + expected)

algorithm = CancerDiagnosis()
algorithm.train_neural_network("/Users/student/Desktop/Cancer Data/cancer_data.csv")
algorithm.test_neural_network("/Users/student/Desktop/Cancer Data/cancer_data.csv")
