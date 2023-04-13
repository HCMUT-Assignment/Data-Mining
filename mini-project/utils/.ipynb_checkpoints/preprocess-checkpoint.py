import numpy as np
# from sklearn 
class Preprocessing:

    def __init__(self):
        pass

    def apply_pipeline(self, input):
        pass
    
    def apply_minmax_scale(self, input):

        samples, n_feature = input.shape
        max_scale           = np.max(input, axis = 0, keepdims = True)
        min_scale           = np.min(input, axis = 0, keepdims = True)
        assert max_scale.shape[1] == n_feature, "Error in preserving the number of features"
        scale               = (input - min_scale) / (max_scale - min_scale)
        return scale, (max_scale, min_scale)

    def apply_log_transform(self, input):
        pass

    def apply_rolling_means(self, input):
        pass

    def apply_handle_nan(self, input):
        pass

    def apply_handle_sth(self, input):
        pass