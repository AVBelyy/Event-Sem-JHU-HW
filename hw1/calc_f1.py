import sys
import json
from sklearn.metrics import f1_score, classification_report


if __name__ == '__main__':
    pred_file = sys.argv[1]
    y_true, y_pred = [], []
    label_map = {'positive': 1, 'negative': 0}

    with open(pred_file) as fin:
        for line in fin:
            item = json.loads(line)
            y_true.append(label_map[item['true_labels']])
            y_pred.append(label_map[item['pred_labels']])

    print(classification_report(y_true, y_pred))
