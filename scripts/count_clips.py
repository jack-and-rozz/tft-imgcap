
import glob
from collections import Counter
from pprint import pprint

paths = [l.split('/')[-1].split('.')[0] for l in glob.glob('datasets/clipped/*.png')]

pprint(Counter(paths))


