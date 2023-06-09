from torch.optim import Adam, SGD
from sklearn.metrics import recall_score, fbeta_score

from dataset import DCNNDataset, LSTMDataset
from model import ReducedInceptionResNet, LSTMClassifier

def evaluate(label, pred):
    '''
    metric1 = FNR
    metric2 = ER
    metric3 = F-beta(1.5)
    '''
    fnr = 1 - recall_score(label, pred)
    er = 1 - ((label == pred).sum() / label.shape[0])
    fbeta = fbeta_score(label, pred, beta=1.5)
    return fnr, er, fbeta

def get_model(model_type, args):
    if model_type.lower() == 'dcnn':
        return ReducedInceptionResNet()
    if model_type.lower() == 'lstm':
        return LSTMClassifier(args.hid_dim)
    else:
        raise NotImplementedError('No model type')

def get_optimizer(optimizer_type, model, lr):
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr)
    if optimizer_type.lower() == 'sgd':
        return SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('No optimizer type')

def get_dataset(data_path, model_type, window_size):
    if model_type.lower() == 'dcnn':
        return DCNNDataset(data_path, window_size)
    if model_type.lower() == 'lstm':
        return LSTMDataset(data_path, window_size)
    else:
        raise NotImplementedError('No dataset for model type')