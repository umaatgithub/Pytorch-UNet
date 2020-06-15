import yaml

config_file_path = "/UNet/src/config/config-drone.yaml"

# The YAML parsers creates a dictionary that links the variables and the values
# The data can then be asked as follows data['id1']['subid1']
config_file = open(config_file_path, 'r')
data = yaml.load(config_file, Loader=yaml.FullLoader)

if __name__ == "__main__":

    print('Input channels : ', data['network']['input']['channels'])

    print('Output channels : ', data['network']['output']['channels'])
    print('Output label 1 : ', data['class']['labels'][0])
    print('Output color 1 : ', data['class']['colors'][0])

