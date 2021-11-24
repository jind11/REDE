from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve, f1_score, precision_score, recall_score
from inspect import signature
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class oodMetrics():
    def __init__(self):
        #dictionary to collect metric names
        self.metrics={}

    def resetMetrics():
        self.metrics={}

    def compute_all_metrics(self,y_gt,y_pred,choose_thresholds=True, fixed_threshold=None):
        precision, recall, pr_thresholds = precision_recall_curve(y_gt, y_pred)
        average_precision = average_precision_score(y_gt, y_pred)
        f1_scores = [2*x*y/(x+y) for x,y in zip(precision,recall)]

        precision_reverse, recall_reverse, pr_reverse_thresholds =  precision_recall_curve(1-y_gt, -y_pred)

        fpr, tpr, thresholds = roc_curve(y_gt, y_pred)
        fpr_reverse, tpr_reverse, thresholds_reverse = roc_curve(1 - y_gt, -y_pred)
        auc_area = auc(fpr,tpr)
        aupr_out_area = auc(recall,precision)
        aupr_in_area = auc(recall_reverse, precision_reverse)

        precision_at_half, recall_at_half, f1_at_half = 0,0,0
        min_distance_to_half = 1
        for thIndex,threshold in enumerate(pr_thresholds):
            if abs(threshold-0.5)<=min_distance_to_half:
                min_distance_to_half = abs(threshold-0.5)
                precision_at_half = precision[thIndex]
                recall_at_half = recall[thIndex]
                f1_at_half = f1_scores[thIndex]
        # print(pr_thresholds)
        self.metrics["PR@0.5"] = precision_at_half
        self.metrics["REC@0.5"] = recall_at_half
        self.metrics["F1@0.5"] = f1_at_half

        if choose_thresholds:
            fpr_at_95_tpr, fpr_at_80_tpr, fpr_id_at_95_tpr = 1.0, 1.0, 1.0
            fpr_at_95_tpr_thresh, fpr_at_80_tpr_thresh = max(thresholds), max(thresholds)
            for f,t,th in zip(fpr,tpr,thresholds):
                if t>=0.95:
                    fpr_at_95_tpr = f
                    fpr_at_95_tpr_thresh = th
                    break
            for f,t,th in zip(fpr,tpr,thresholds):
                if t>=0.80:
                    fpr_at_80_tpr = f
                    fpr_at_80_tpr_thresh = th
                    break
            recall_at_95_pr, recall_at_80_pr = 0.0 , 0.0
            recall_at_95_pr_thresh, recall_at_95_pr_thresh = 1.0 , 1.0
            for r, p , prth in zip(recall,precision,pr_thresholds):
                if p>=0.95:
                    recall_at_95_pr = r
                    recall_at_95_pr_thresh = prth
                    break
            for r, p, prth in zip(recall,precision,pr_thresholds):
                if p>=0.80:
                    recall_at_80_pr = r
                    recall_at_80_pr_thresh = prth
                    break

            for f, t, th in zip(fpr_reverse, tpr_reverse, thresholds_reverse):
                if t >= 0.95:
                    fpr_id_at_95_tpr = f
                    fpr_id_at_95_tpr_thresh = th
                    break

        elif fixed_threshold!=None:
            fpr_at_95_tpr, fpr_at_80_tpr = 1.0, 1.0
            fpr_at_95_tpr_thresh, fpr_at_80_tpr_thresh = max(thresholds), max(thresholds)
            for f,t,th in zip(fpr,tpr,thresholds):
                if th==fixed_threshold:
                    fpr_at_95_tpr = f
                    fpr_at_95_tpr_thresh = th
                    break
            for f,t,th in zip(fpr,tpr,thresholds):
                if th==fixed_threshold:
                    fpr_at_80_tpr = f
                    fpr_at_80_tpr_thresh = th
                    break
            recall_at_95_pr, recall_at_80_pr = 0.0 , 0.0
            recall_at_95_pr_thresh, recall_at_95_pr_thresh = 1.0 , 1.0
            for r, p , prth in zip(recall,precision,pr_thresholds):
                if prth==fixed_threshold:
                    recall_at_95_pr = r
                    recall_at_95_pr_thresh = prth
                    break
            for r, p, prth in zip(recall,precision,pr_thresholds):
                if prth==fixed_threshold:
                    recall_at_80_pr = r
                    recall_at_80_pr_thresh = prth
                    break

        pr_threshold_highest_f1 = pr_thresholds[f1_scores.index(max(f1_scores))]
        if fixed_threshold is None:
            f1_score_ood = max(f1_scores)
            assert f1_score_ood == f1_score(y_gt, y_pred >= pr_threshold_highest_f1)
            precision_score_ood = precision_score(y_gt, y_pred >= pr_threshold_highest_f1)
            recall_score_ood = recall_score(y_gt, y_pred >= pr_threshold_highest_f1)
        else:
            f1_score_ood = f1_score(y_gt, y_pred >= fixed_threshold)
            precision_score_ood = precision_score(y_gt, y_pred >= fixed_threshold)
            recall_score_ood = recall_score(y_gt, y_pred >= fixed_threshold)


        self.metrics["recall"] = recall
        self.metrics["precision"] = precision
        self.metrics["pr_thresholds"] = pr_thresholds
        self.metrics["average_precision"] = average_precision
        self.metrics["recall_reverse"] = recall_reverse
        self.metrics["precision_reverse"] = precision_reverse
        self.metrics["pr_thresholds_reverse"] = pr_reverse_thresholds
        self.metrics["f1_scores"] = f1_scores
        self.metrics["fpr"] = fpr
        self.metrics["tpr"] = tpr
        self.metrics["thresholds"] = thresholds

        self.metrics["FPR_OOD@80%TPR"] = fpr_at_80_tpr
        self.metrics["FPR_OOD@95%TPR"] = fpr_at_95_tpr
        self.metrics["REC_OOD@80%PR"] = recall_at_80_pr
        self.metrics["REC_OOD@95%PR"] = recall_at_95_pr
        self.metrics["FPR_ID@95%TPR"] = fpr_id_at_95_tpr
        self.metrics["AUROC"] = auc_area
        self.metrics["AUPRIn"] = aupr_in_area
        self.metrics["AUPROut"] = aupr_out_area
        self.metrics["F1 Scores OOD"] = f1_score_ood
        self.metrics["Max F1 Scores OOD"] = max(f1_scores)
        self.metrics["Threshold of Max F1 Scores OOD"] = pr_thresholds[f1_scores.index(max(f1_scores))]
        self.metrics["P Scores OOD"] = precision_score_ood
        self.metrics["R Scores OOD"] = recall_score_ood

        return pr_threshold_highest_f1


    def plot_PR_curve(self,plotName):
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(self.metrics["recall"], self.metrics["precision"], color='b', alpha=0.2, where='post')
        plt.fill_between(self.metrics["recall"], self.metrics["precision"], alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0,1.05])
        plt.xlim([0.0,1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(self.metrics["average_precision"]))
        plt.savefig(plotName)


    def pretty_print_metrics(self, dump_metrics_file_path=None):
        print("-----OOD.METRICS.SUMMARY.BEGIN-----")
        keyList = ["PR@0.5","REC@0.5","F1@0.5","FPR_OOD@80%TPR","FPR_OOD@95%TPR","REC_OOD@80%PR","REC_OOD@95%PR","FPR_ID@95%TPR","AUROC","AUPRIn","AUPROut","F1 Scores OOD", "P Scores OOD", "R Scores OOD", "Max F1 Scores OOD", "Threshold of Max F1 Scores OOD"]
        dump_metrics = {}
        for key in keyList:
            print(key,":",self.metrics[key])
            dump_metrics[key] = self.metrics[key]
        print("-----OOD.METRICS.SUMMARY.END-----")
        if dump_metrics_file_path is not None:
            json.dump(dump_metrics, open(dump_metrics_file_path, 'w'))

        return dump_metrics

