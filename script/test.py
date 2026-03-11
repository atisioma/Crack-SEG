from crack import CrackTrainTool


def crack_train():
    tool = CrackTrainTool('../config/config.yaml')
    tool.run()



if __name__ == '__main__':
    crack_train()
