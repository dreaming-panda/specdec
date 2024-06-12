import yaml

f = open("layer_sparse.yaml")

x = yaml.safe_load(f)
print(x)