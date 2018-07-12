from tqdm import tqdm
import keras.backend as K


def get_gpu_session(ratio=None, interactive=False):
    config = K.tf.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = K.tf.InteractiveSession(config=config)
    else:
        sess = K.tf.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)


class WeightConverter:

    def __init__(self, keras_model=None, weight_dict=None):
        if keras_model is None:
            self.weight_dict = weight_dict
        elif weight_dict is None:
            self.weight_dict = dict()
            for weight_var in tqdm(keras_model.weights):
                self.weight_dict[weight_var.name] = K.eval(weight_var)
        else:
            raise ValueError

    def replace_depthwise_weight(self, query, param):
        target = query.replace("_weight", '/depthwise_kernel:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight.transpose(2, 3, 0, 1))

    def replace_weight(self, query, param):
        target = query.replace("_weight", '/kernel:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight.transpose(3, 2, 0, 1))

    def replace_bias(self, query, param):
        target = query.replace("_bias", '/bias:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight)

    def replace_gamma(self, query, param):
        target = query.replace("_gamma", '/gamma:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight)

    def replace_beta(self, query, param):
        target = query.replace("_beta", '/beta:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight)

    def replace_running_mean(self, query, param):
        target = query.replace("_running_mean", '/moving_mean:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight)

    def replace_running_var(self, query, param):
        target = query.replace("_running_var", '/moving_variance:0')
        assert target in self.weight_dict, "%s->%s" % (query, target)
        weight = self.weight_dict[target]
        param.set_data(weight)

    def set_parameters(self, gluon_model):
        for k, param in gluon_model.collect_params().items():
            if k.endswith("depthwise_weight"):
                self.replace_depthwise_weight(k, param)
            elif k.endswith("weight"):
                self.replace_weight(k, param)
            elif k.endswith("bias"):
                self.replace_bias(k, param)
            elif k.endswith("gamma"):
                self.replace_gamma(k, param)
            elif k.endswith("beta"):
                self.replace_beta(k, param)
            elif k.endswith("running_mean"):
                self.replace_running_mean(k, param)
            elif k.endswith("running_var"):
                self.replace_running_var(k, param)
            else:
                raise ValueError
