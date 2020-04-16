
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = sys.argv[1]

    plt.figure(figsize=(12.0, 12.0))

    with open(file, encoding="utf-8", newline="\n") as csv:
        loss = list((float(row.strip().split(',')[0]) for row in (csv) if len(row) > 0))

    loss = np.stack(loss)

    path = os.path.dirname(file).split('\\')[-1]

    plt.plot(loss)
    plt.title(path)
    
    plt.xlabel('batch')
    plt.show(block=False)
    plt.ylabel('loss')

    plt.waitforbuttonpress()

    dir = os.path.dirname(os.path.abspath(file))

    plt.savefig('%s/%s_plot.jpg' % (dir,path), bbox_inches='tight')


