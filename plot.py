import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd


def CM_calculator(true_class, predict_class, batch_size, result_path, mode):
    target_true = true_class.cpu()
    if mode == 'Bert':
        target_predicted = predict_class.cpu()
    else:
        target_predicted = predict_class.resize_(batch_size).cpu()
    conf_matrix = confusion_matrix(target_true, target_predicted)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Negative', 'Natural', 'Positive'],
                                  columns=['Negative', 'Natural', 'Positive'])
    plt.figure(figsize=(8,7), dpi=100)
    sb.set(font_scale=1.5)  # for label size
    sb.heatmap(conf_matrix_df, annot=True, annot_kws={"size": 14}, cmap=plt.cm.Reds, fmt="d") # font size
    plt.ylabel('True Classes')
    plt.xlabel('Prediction Classes')
    plt.title('Confusion Matrix for Sentiment140 Classification', color='darkblue')
    plt.savefig(result_path+'CM_{0}.jpg'.format(mode), dpi=200)
    # plt.show()


def show_plot(train_loss, test_loss, train_acc, test_acc, result_path, mode):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Train vs Test Loss', color='darkblue')
    plt.plot(train_loss, color='blue', label='Train Loss')
    plt.plot(test_loss, color='orange', label='Test Loss')
    plt.legend()
    plt.savefig(result_path+'Loss_{0}.jpg'.format(mode), dpi=200)
    # plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Accuracy During Time Training', color='darkblue')
    plt.plot(train_acc, color='blue', label='Train Acc')
    plt.plot(test_acc, color='orange', label='Test Acc')
    plt.legend()
    plt.savefig(result_path+'Acc_{0}.jpg'.format(mode), dpi=200)
    # plt.show()
