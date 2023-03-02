
class Options():
    def __init__(self):
        super().__init__()
        # 超参数
        # 超参数
        self.Seed = 1234  # 随机种子
        self.Epoch = 100
        self.Learning_Rate = 1e-3  # 每个阶段学习率从头开始
        self.Batch_Size_Train = 18
        self.Batch_Size_Val = 18
        self.Patch_Size_Train = 128
        self.Patch_Size_Val = 128
        self.Warmup_Epochs = 3  # 预热训练次数
        # 训练集路径
        self.Input_Path_Train = 'E://allweather/input/'
        self.Target_Path_Train = 'E://allweather/gt/'
        # 验证集路径
        self.Input_Path_Val = 'E://allweather_test/raindrop_a/input/'
        self.Target_Path_Val = 'E://allweather_test/raindrop_a/target/'
        # 测试集路径
        flag = 4
        dict = {
            0: 'Snow100k_L',
            1: 'Snow100k_M',
            2: 'Snow100k_S',
            3: 'raindrop_a',
            4: 'Test1',
        }
        choice = dict[flag]

        # self.Input_Path_Test = 'E://allweather_test/' + choice + '/input/'
        # self.Target_Path_Test = 'E://allweather_test/' + choice + '/target/'
        # self.Result_Path_Test = 'E://allweather_test/' + choice + '/result_Uformer/'

        self.Input_Path_Test = 'E://allweather_real_test/realsnow/input/'
        self.Result_Path_Test = 'E://allweather_real_test/realsnow/result_Uformer/'


        self.MODEL_SAVE_PATH = './'

        # 线程数
        self.Num_Works = 4
        # 是否使用 CUDA
        self.CUDA_USE = True