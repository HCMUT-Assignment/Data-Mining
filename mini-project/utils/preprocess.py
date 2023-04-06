import numpy as np
# from sklearn 
class Preprocessing:

    def __init__(self):
        pass

    def apply_pipeline(self, input):
        pass
    
    def apply_minmax_scale(self, input):

        n_features, samples = input.shape
        max_scale           = np.max(input, axis = 1, keepdims= True)
        assert max_scale.shape[0] == n_features, "Error in preserving the number of features"
        return input / max_scale

    def apply_log_transform(self, input):
        pass

    def apply_rolling_means(self, input):
        pass

    def apply_handle_nan(self, input):
        pass

    def apply_handle_sth(self, input):
        pass