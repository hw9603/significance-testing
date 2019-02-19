import numpy as np
from numpy import genfromtxt
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
from scipy import stats
import csv


class SignificanceTesting(object):
	def __init__(self, file_path):
		self.filePath = file_path
		self.models_scores = None
		self.data = None
		self.load_data()

		# statistical analysis
		self.mean = 0.0
		self.median = 0.0
		self.mode = 0.0
		self.min = 0.0
		self.max = 0.0

		# table data entry
		w = 6
		h = 13
		self.results_data = [[0 for x in range(w)] for y in range(h)]

	def load_data(self):
		self.models_scores = ['Baseline_R2', 'Baseline+Fusion_R2', 'Baseline+Ordering_R2',
							'Baseline+Ordering+Fusion_R2', 'Baseline_RSU4',	'Baseline+Fusion_RSU4',
							'Baseline+Ordering_RSU4', 'Baseline+Ordering+Fusion_RSU4']
		self.data = genfromtxt(self.filePath, delimiter=',')[1:].T

	def ks_test(self, list_a, list_b):
		value, pvalue = ks_2samp(list_a, list_b)
		return pvalue

	def t_test(self, list_a, list_b):
		value, pvalue = ttest_ind(list_a, list_b)
		return pvalue

	def wilcoxon_test(self, list_a, list_b):
		T, pvalue = wilcoxon(list_a, list_b)
		return pvalue

	def stat_analysis(self):
		# print(self.data)
		self.mean = np.mean(self.data, axis=1)
		# [ 0.4165538  0.3029308  0.4124338  0.2929442  0.4165538  0.3029308  0.4124338  0.2929442]
		self.median = np.median(self.data, axis=1)
		# [ 0.33333   0.218255  0.318015  0.207615  0.33333   0.218255  0.318015 0.207615]
		self.mode = stats.mode(self.data, axis=1)
		# ModeResult(mode=array([[ 1.], [ 0.], [ 0.], [ 0.], [ 1.], [ 0.], [ 0.], [ 0.]]),
		# 			 count=array([[12], [17], [19], [17], [12], [17], [19], [17]]))
		self.min = np.min(self.data, axis=1)  # [ 0.  0.  0.  0.  0.  0.  0.  0.]
		self.max = np.max(self.data, axis=1)  # [ 1.  1.  1.  1.  1.  1.  1.  1.]

	def mean_diff_test(self, a, b):
		return b - a

	def init_table(self):
		self.results_data[0] = ['metric', 'model', 'mean diff', 'P(T test)', 'P(wilcoxon test)', 'P(ks test)']
		for row in xrange(1, 7):
			self.results_data[row][0] = 'ROUGE-2'
		for row in xrange(7, 13):
			self.results_data[row][0] = 'ROUGE-SU4'
		self.results_data[1][1] = 'Baseline & Fusion'
		self.results_data[2][1] = 'Baseline & Ordering'
		self.results_data[3][1] = 'Baseline & Ordering+Fusion'
		self.results_data[4][1] = 'Fusion & Ordering'
		self.results_data[5][1] = 'Fusion & Ordering+Fusion'
		self.results_data[6][1] = 'Ordering & Ordering+Fusion'

		self.results_data[7][1] = 'Baseline & Fusion'
		self.results_data[8][1] = 'Baseline & Ordering'
		self.results_data[9][1] = 'Baseline & Ordering+Fusion'
		self.results_data[10][1] = 'Fusion & Ordering'
		self.results_data[11][1] = 'Fusion & Ordering+Fusion'
		self.results_data[12][1] = 'Ordering & Ordering+Fusion'

		return self.results_data

	def fill_results(self):
		# mean diff
		# baseline/fusion, baseline/ordering, baseline/ordering+fusion,
		# fusion/ordering, fusion/ordering+fusion, ordering/ordering+fusion
		mean_rouge2 = self.mean[0:4]
		mean_rougesu4 = self.mean[4:]
		assert(len(mean_rouge2) == len(mean_rougesu4))
		self.cross_evaluate(self.mean_diff_test, 2, mean_rouge2, mean_rougesu4)

		data_rouge2 = self.data[0:4]
		data_rouge4 = self.data[4:]
		self.cross_evaluate(self.t_test, 3, data_rouge2, data_rouge4)
		self.cross_evaluate(self.wilcoxon_test, 4, data_rouge2, data_rouge4)
		self.cross_evaluate(self.ks_test, 5, data_rouge2, data_rouge4)

	def cross_evaluate(self, fn, col_id, data_rouge2, data_rouge4):
		index = 1
		for i, list_a in enumerate(data_rouge2):
			for j, list_b in enumerate(data_rouge2[i + 1:]):
				self.results_data[index][col_id] = fn(list_a, list_b)
				index += 1
		for i, list_a in enumerate(data_rouge4):
			for j, list_b in enumerate(data_rouge4[i + 1:]):
				self.results_data[index][col_id] = fn(list_a, list_b)
				index += 1

	def write_output(self):
		results_file = open('SigTestResults2.csv', 'w')
		with results_file:
			writer = csv.writer(results_file)
			writer.writerows(self.results_data)
		results_file.close()
		print("Dump result to SigTestResults2.csv successfully!")

	# def boxing_plot(self):
	# 	plt.figure()
	# 	plt.boxplot(self.data.T)
	# 	plt.show()


if __name__ == '__main__':
	file_path = "ROUGE_SCORES.csv"
	sigInstance = SignificanceTesting(file_path)
	sigInstance.stat_analysis()
	# ks = sigInstance.ks_test()
	# t = sigInstance.t_test()
	# w = sigInstance.wilcoxon_test()
	sigInstance.init_table()
	sigInstance.fill_results()
	sigInstance.write_output()


