"""
This script handles everything that happens in the 'config.properties' file
"""


# %% Class Config
class Config(object):
    """
    This class will handle all the aspects relating to the configurations needed in the application
    """

    def __init__(self):
        self.config = {}
        self.__init_config__()

    def __init_config__(self):
        """
        This method gets all the config from the 'config.properties' file and saves it in the local 'self.config'
        parameter

        :return:
        """
        try:
            separator = '='

            with open('config.properties') as f:
                for line in f:
                    if separator in line:
                        # Find the name and value by splitting the string
                        [name, value] = line.split(separator, 1)

                        # Assign key value pair to dict
                        # strip() removes white space from the ends of strings

                        value = str(value).replace('\n', '')
                        if not str(value).replace('.', '', 1).isdigit():
                            if str(value).lower() == 'true':
                                self.config[name.strip()] = True
                            elif str(value).lower() == 'false':
                                self.config[name.strip()] = False
                            else:
                                self.config[name.strip()] = value.strip()
                        else:
                            if str(value).isdigit():
                                self.config[name.strip()] = int(value)
                            else:
                                self.config[name.strip()] = float(value)
            return True

        except Exception as error_message:
            raise Exception(error_message)

    def get(self, key):
        """
        This method returns the configuration value for a given key

        :param key: (String) The key of the desired config
        :return: (String or Number) The requested value
        """
        try:
            if not isinstance(key, str):
                raise ValueError('Invalid key %s type. It should be a string' % str(key))
            elif key not in self.config.keys():
                raise ValueError('Key %s not found.' % str(key))

            return self.config[key]
        except Exception as error_message:
            raise Exception(error_message)


config = Config()
