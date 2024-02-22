// smelu.h

#include <vector>

template <typename T>
class SmeLU {
public:
    explicit SmeLU(T alpha_value = T(1.0), std::size_t size = 1) : alpha(size, alpha_value) {}
    std::vector<T> forward(const std::vector<T>& input);
    std::vector<T> backward(const std::vector<T>& input, const std::vector<T>& grad_output);

private:
    std::vector<T> alpha;
};

template <typename T>
std::vector<T> SmeLU<T>::forward(const std::vector<T>& input) {
    std::vector<T> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha[i % alpha.size()] * (std::exp(input[i]) - T(1.0));
    }
    return output;
}

template <typename T>
std::vector<T> SmeLU<T>::backward(const std::vector<T>& input, const std::vector<T>& grad_output) {
    std::vector<T> grad_input(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        grad_input[i] = input[i] > 0 ? grad_output[i] : grad_output[i] * alpha[i % alpha.size()] * std::exp(input[i]);
    }
    return grad_input;
}
