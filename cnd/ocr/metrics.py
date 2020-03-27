from argus.metrics import Metric
from cnd.ocr.converter import strLabelConverter


class StringAccuracy(Metric):
    name = "str_accuracy"
    better = "max"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.correct = 0
        self.count = 0

    def compare_two_str(self, a, b):
        eq_means = 0
        for pair in zip(a, b):
            if pair[0] == pair[1]:
                eq_means += 1
        return eq_means

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        for i in range(min(len(preds), len(targets))):
            self.correct += self.compare_two_str(preds[i], targets[i])
        
        for target in targets[:min(len(preds), len(targets))]:
            self.count += len(target)

    def compute(self):
        if self.count == 0:
            return 0
        return self.correct / self.count
