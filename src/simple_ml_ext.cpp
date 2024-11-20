#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

template<class T>
T get_from_2d_array(T* arr, int col_size, int row_idx, int col_idx){
  return arr[row_idx*col_size + col_idx];
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t n_batch = m / batch;
    if (m % batch != 0) {
        n_batch += 1;
    }
    for (size_t i = 0; i < n_batch; ++i) {
        std::vector<std::vector<float>> inter_array(batch, std::vector<float>(k, 0));
        size_t minibatch = 0;
        for (; minibatch < batch && minibatch + i * batch < m; ++minibatch) {
            size_t idx = i * batch + minibatch;
            float sum = 0;
            for (size_t col=0; col < k; ++col) {
                for (size_t j=0; j < n; ++j) {
                    inter_array[minibatch][col] += get_from_2d_array(X, n, idx, j) * get_from_2d_array(theta, k, j, col);
                }

                inter_array[minibatch][col] = std::exp(inter_array[minibatch][col]);
                sum += inter_array[minibatch][col];
            }

            // normalize_softmax and minus one-hot
            for (size_t col=0; col < k; ++col) {
                inter_array[minibatch][col] /= sum;
                if (col == y[minibatch + i * batch]) {
                    inter_array[minibatch][col] -= 1;
                }
            }
        }

        // calculate gradient
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < k; ++col) {
                float tmp = 0;
                for (size_t j = 0; j < minibatch; ++j) {
                    tmp += get_from_2d_array(X, n, i * batch + j, row) * inter_array[j][col];
                }
                tmp = tmp / minibatch;
                theta[row * k + col] -= lr * tmp;
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
