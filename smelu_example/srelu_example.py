from smelu import SmeLUf, SmeLUD

# create SmeLU instances for float and double data types with custom alpha value
smelu_f = SmeLUf(alpha_value=0.9)
smelu_d = SmeLUD(alpha_value=0.9)

# example input data
input_data = [1.0, -2.0, 3.0, -4.0]

# compute forward pass for SmeLU with float data type
output_f = smelu_f.forward(input_data)

# compute forward pass for SmeLU with double data type
output_d = smelu_d.forward(input_data)

# print output of forward pass
print("SmeLU forward pass for float data type:", output_f)
print("SmeLU forward pass for double data type:", output_d)

# example gradient output data
grad_output_data = [1.0, 1.0, 1.0, 1.0]

# compute backward pass for SmeLU with float data type
grad_input_f = smelu_f.backward(input_data, grad_output_data)

# compute backward pass for SmeLU with double data type
grad_input_d = smelu_d.backward(input_data, grad_output_data)

# print gradient inputs
print("SmeLU backward pass grad input for float data type:", grad_input_f)
print("SmeLU backward pass grad input for double data type:", grad_input_d)
