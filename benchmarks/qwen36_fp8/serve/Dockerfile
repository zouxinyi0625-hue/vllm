# cuda12.9 vllm 0.23.0 91df0fad4
FROM f9309c3acdd842848c88032e1ec736d2.azurecr.io/azureml/azureml_286dce7ac18ccbce937cad4e6dd67e72
WORKDIR /workspace
COPY . /workspace/
RUN find /workspace -name "*.sh" -exec sed -i 's/\r$//' {} +
