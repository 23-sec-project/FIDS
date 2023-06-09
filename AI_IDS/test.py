import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from util import get_model, get_dataset, evaluate


def test(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    model.eval()
    tbar = tqdm(dataloader)
    label, pred = None, None
    tbar.set_description(f'Testing')
    for x, y in tbar:
        x = x.to(device)

        res = model(x)
        res = (res > 0.5).detach().cpu().to_numpy()
        if label is None:
            label = np.array(y.to_numpy())
            pred = np.array(res)
        else:
            label = np.append(label, y.to_numpy())
            pred = np.append(pred, res)
    print(evaluate(label, pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_type', default='dcnn', type=str, help='model type')
    parser.add_argument('--window_size', default=3, type=int, help='window size')
    
    # for lstm
    parser.add_argument('--hid_dim', default=512, type=int, help='hidden dimension')
    
    # else
    parser.add_argument('--save_path', default='./model.pth', type=str, help='model save path')
    parser.add_argument('--train_f_path', default='./train.csv', type=str, help='train file path')
    parser.add_argument('--test_f_path', default='./test.csv', type=str, help='test file path')
    
    args = parser.parse_args()
    model = get_model(args.model_type, args)
    model.load_state_dict(torch.load(args.save_path))
    test_data = get_dataset(args.train_f_path, args.model_type, args.window_size)
    test_dataloader = DataLoader(test_data, args.batch_size, False)
    test(model, test_dataloader)
    