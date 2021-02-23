import os, sys
import matplotlib.pyplot as plt
import re
import numpy as np

def main():
	if len(sys.argv) != 2:
		print('usage: python %s <log file location>' % sys.argv[0])

	log_file = sys.argv[1]

	dir_title = os.path.dirname(os.path.realpath(log_file)).split("\\")[-1]

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
				if z:
					lst = eval_data.get(z.groups()[0], [])
					lst.append(float(z.groups()[1]))
					eval_data[z.groups()[0]] = lst
			else:
				z = re.match(re_train, line)
				if z:
					lst = train_data.get(z.groups()[0], [])
					lst.append(float(z.groups()[1]))
					train_data[z.groups()[0]] = lst

	plt.rcParams["figure.figsize"] = (20,15)
	for key, train_plot in train_data.items():
		eval_plot = eval_data[key]

		data_train = np.array(train_plot[10:])
		data_eval = np.array(eval_plot)

		data_eval = np.repeat(data_eval, int(data_train.shape[0] / data_eval.shape[0]))

		plt.plot(data_train, label='train')
		plt.plot(data_eval, label='eval')

		plt.title('Eval losses: ' + key + " " + dir_title)
		plt.legend()	
		file_out = os.path.join(os.path.dirname(os.path.realpath(log_file)), 'Loss_' + key +'.png')
		plt.savefig(file_out)
		plt.close()



if __name__ == "__main__":
	main()
