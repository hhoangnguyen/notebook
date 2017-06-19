"""
Equivalent python code for TensorFlow Core code
Reference: https://www.tensorflow.org/get_started/get_started
"""

# setup variables
x_array = [1, 2, 3, 4]
y_array = [0, -1, -2, -3]
W = .3
b = -.3


def number_format(number):
    return round(number, 2)


def adder_node(a, b):
    return a + b


def add_and_triple(add_function, a, b):
    return add_function(a, b) * 3

print(add_and_triple(adder_node, 3, 4.5)) # 22.5


def linear_model(W, b, x):
    return number_format(W * x + b)


def run_linear_model(W, b, x_array):
    y_array = []
    for x in x_array:
        y_array.append(linear_model(W, b, x))
    return y_array

# calculate predicted y
print(run_linear_model(W, b, x_array)) # [0, 0.3, 0.6, 0.9]


def squared_deltas(predicted, actual):
    squared_deltas_total = []
    for i in range(len(predicted)):
        squared_deltas_total.append((predicted[i] - actual[i])**2)
    return squared_deltas_total


def reduce_sum(array):
    r_sum = 0
    for item in array:
        r_sum += item
    return number_format(r_sum)

# cost function
print(reduce_sum(squared_deltas(run_linear_model(W, b, x_array), y_array))) # 23.66

# assign W and b to optimal
print(reduce_sum(squared_deltas(run_linear_model(-1.0, 1.0, x_array), y_array))) # 0.00

"""
The tutorial finishes with an optimizer, Gradient Descent
See here for equivalent python code: https://github.com/hhoangnguyen/notebook/tree/master/linear_regression
"""
