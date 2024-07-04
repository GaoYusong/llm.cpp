# A tool created by ChatGPT to generate PyTorch tensors and perform operations, then print them as C++ 1D float arrays.

import torch
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Generate PyTorch tensors and perform operations, then print them as C++ 1D float arrays.")
parser.add_argument("expressions", type=str, nargs='+', help='Tensor expressions like "a(2, 3, 2)" "b(2, 4, 2)" "c(2, 3, 4)" "a @ b + c"')

# Parse the command-line arguments
args = parser.parse_args()

# Function to parse tensor definitions and operations
def parse_expression(expression):
    if '(' in expression:
        name, dims = expression.split('(')
        dims = tuple(map(int, dims.rstrip(')').split(',')))
        return name, dims
    return expression

# Create tensors
tensors = {}
dimensions = {}
for expr in args.expressions:
    parsed = parse_expression(expr)
    if isinstance(parsed, tuple):
        name, dims = parsed
        tensors[name] = torch.rand(dims, requires_grad=True) * 2 - 1  # Random values between -1 and 1
        tensors[name].retain_grad()
        dimensions[name] = dims

# Perform operations
def evaluate_operation(operation):
    if '@' in operation and '+' in operation:
        left, rest = operation.split('@')
        right, addend = rest.split('+')
        left = left.strip()
        right = right.strip()
        addend = addend.strip()
        result = torch.matmul(tensors[left], tensors[right].transpose(-2, -1)) + tensors[addend]
    elif '+' in operation:
        left, right = operation.split('+')
        left = left.strip()
        right = right.strip()
        result = tensors[left] + tensors[right]
    elif '-' in operation:
        left, right = operation.split('-')
        left = left.strip()
        right = right.strip()
        result = tensors[left] - tensors[right]
    elif '*' in operation:
        left, right = operation.split('*')
        left = left.strip()
        right = right.strip()
        result = tensors[left] * tensors[right]
    elif '/' in operation:
        left, right = operation.split('/')
        left = left.strip()
        right = right.strip()
        result = tensors[left] / tensors[right]
    elif '@' in operation:
        left, right = operation.split('@')
        left = left.strip()
        right = right.strip()
        result = torch.matmul(tensors[left], tensors[right].transpose(-2, -1))
    result.retain_grad()
    return result

operations = [expr for expr in args.expressions if any(op in expr for op in ['+', '-', '*', '/', '@'])]
for operation in operations:
    result = evaluate_operation(operation)
    result_name = "result"
    tensors[result_name] = result
    dimensions[result_name] = result.shape

# Perform backward pass
tensors["result"].sum().backward()

# Print tensors and gradients as C++ 1D float arrays
for name, tensor in tensors.items():
    tensor_1d = tensor.view(-1).detach().numpy()
    dims = dimensions[name]
    if name == "result":
        print(f"std::vector<int> result_dims = {{{', '.join(map(str, dims))}}};")
        print(f"float result_data[] = {{")
    else:
        print(f"std::vector<int> {name}_dims = {{{', '.join(map(str, dims))}}};")
        print(f"float {name}_data[] = {{")
    print(", ".join(f"{x:.6f}f" for x in tensor_1d))
    print("};")

    # Print gradients if available
    if tensor.grad is not None:
        grad_1d = tensor.grad.view(-1).detach().numpy()
        print(f"float {name}_grad_data[] = {{")
        print(", ".join(f"{x:.6f}f" for x in grad_1d))
        print("};")
