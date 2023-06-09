import torch

linear = torch.nn.Linear(in_features=5, out_features=10)
inputs = torch.randn(3, 5)
outputs = linear(inputs)

name_to_tensor = {
    "linear_weight": linear.weight,
    "linear_bias": linear.bias,
    "linear_inputs": inputs,
    "linear_outputs": outputs,
}

for name, tensor in name_to_tensor.items():
    with open(f"models/test/{name}", "wb") as file_:
        file_.write(tensor.reshape(-1).detach().numpy().tobytes())
