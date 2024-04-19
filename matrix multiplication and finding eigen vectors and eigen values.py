import tensorflow as tf

print("Matrix Multiplication Demo")

x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print("Matrix x:\n", x)

y = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
print("Matrix y:\n", y)

z = tf.matmul(x, y)
print("Product:\n", z)

e_matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
print("\nMatrix A:\n", e_matrix_A)

eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("\nEigen Vectors:\n", eigen_vectors_A.numpy())
print("\nEigen Values:\n", eigen_values_A.numpy())
