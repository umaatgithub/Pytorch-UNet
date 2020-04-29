import yaml

config_file_path = "config.yaml"

# The YAML parsers creates a dictionary that links the variables and the values
# The data can then be asked as follows data['id1']['subid1']
config_file = open(config_file_path, 'r')
data = yaml.load(config_file)

if __name__ == "__main__":

    print('Input channels : ', data['input']['channels'])

    print('Output channels : ', data['output']['channels'])
    print('Output label 1 : ',data['output']['labels'][0])
    print('Output color 1 : ',data['output']['colors'][0])

