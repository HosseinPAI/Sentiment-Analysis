from dataset import prep_clean_data
import argparse
import os


if __name__ == '__main__':
    # Create Parameters
    parser = argparse.ArgumentParser(description='Pytorch Fast Human Pose Estimation!')
    parser.add_argument('--extract_path', type=str, default='',
                        help='Glove42B_300d path')
    parser.add_argument('--glove42B_path', type=str, default='original_data/glove.42B.300d.zip',
                        help='Glove42B_300d path')
    parser.add_argument('--original_dataset_path', type=str, default='original_data/trainingandtestdata.zip',
                        help='Original Dataset path')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Dataset path')
    parser.add_argument('--mode', type=str, default='LSTM', help='Mode of Training')

    """
        This Function Has Four Modes, You can Set The Mode variable base on following options
        Mode = LSTM     : Simple LSTM with one hidden layer
        Mode = LSTM_Bid : Bidirectional LSTM
        Mode = Pyramid  : Pyramid LSTM
        Mode = Bert     : Pyramid LSTM
    """

    # Start Process
    args = parser.parse_args()

    current_dir = os.getcwd()
    args.extract_path = current_dir + '/original_data/'
    args.dataset_path = current_dir + '/dataset/'
    if not os.path.exists(args.extract_path):
        os.makedirs(args.extract_path)

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    prep_clean_data(args.glove42B_path, args.original_dataset_path, args.extract_path, args.dataset_path, args.mode)
