# cuda12.9 vllm 0.21.1rc1.dev270+g6cbe448ee
# FROM f9309c3acdd842848c88032e1ec736d2.azurecr.io/azureml/azureml_26cb416546ce64c37d203ed6e088be86

# cuda 13.0 vllm 0.22.1rc1.dev288+gd8218b1ee
# FROM f9309c3acdd842848c88032e1ec736d2.azurecr.io/azureml/azureml_18b9a6bea595e6ef57359a3e71719ad8

# cuda12.9 vllm 0.23.0 91df0fad4
FROM f9309c3acdd842848c88032e1ec736d2.azurecr.io/azureml/azureml_286dce7ac18ccbce937cad4e6dd67e72
WORKDIR /workspace
COPY . /workspace/
RUN find /workspace -name "*.sh" -exec sed -i 's/\r$//' {} +
