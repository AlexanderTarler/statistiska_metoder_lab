import numpy as np

# np.ndim is used to get the number of dimensions of an array.

# Since x is an array made up of one dimension (meaning that the array only has one row), the number of dimensions is 1.
x = np.array([1, 2, 3])
x.ndim

# Since y is an array made up of two dimensions (meaning that the array has two rows/arrays), the number of dimensions is 2.
y = np.array([[1, 2, 3], [4, 5, 6]])
y.ndim

print(x.ndim)
print(y.ndim)

# np.ones is used to create an array of ones(1).

x = np.ones(3)
print(x)

y = [[1, 2, 3], [4, 5, 6]]

# np.r_ is used to concatenate the arrays along the first axis. Example:
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = np.r_[x, y]
print(z)

# np.c_ is used to concatenate the arrays along the second axis. Example:

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = np.c_[x, y]
print(z)
