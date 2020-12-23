import os, sys
import matplotlib.pyplot as plt
import re
import numpy as np

def main():
	if len(sys.argv) != 2:
		print('usage: python %s <log file location>' % sys.argv[0])

	log_file = sys.argv[1]
	eval_out = os.path.join(os.path.dirname(os.path.realpath(log_file)), 'Eval_losses.png')
	train_out = os.path.join(os.path.dirname(os.path.realpath(log_file)), 'Train_losses.png')

	assert(os.path.exists(log_file))

	re_eval = r"eval mean (.*): (\d+.\d+)"
	re_train = r"mean (.*): (\d+.\d+)"

	train_data = {}
	eval_data = {}
	with open(log_file, 'r') as f:
		for line in f:
			if 'mean' not in line:
				continue
			if 'ratio' in line:
				continue
			if 'eval' in line:
				z = re.match(re_eval, line)
				lst = eval_data.get(z.groups()[0], [])
				lst.append(float(z.groups()[1]))
				eval_data[z.groups()[0]] = lst
			else:
				z = re.match(re_train, line)
				lst = train_data.get(z.groups()[0], [])
				lst.append(float(z.groups()[1]))
				train_data[z.groups()[0]] = lst

	#plot data
	plt.rcParams["figure.figsize"] = (20,15)
	for key, val in eval_data.items():

		data = np.array(val)
		data /= max(np.max(data), 0.001)

		plt.plot(data, label=key)
	plt.title('Eval losses')
	plt.legend()
	plt.savefig(eval_out)
	plt.close()

	for key, val in train_data.items():
		data = np.array(val)
		data /= max(np.max(data), 0.001)

		plt.plot(data, label=key)
	plt.title('Train losses')
	plt.legend()
	plt.savefig(train_out)
	plt.close()



if __name__ == "__main__":
	main()
