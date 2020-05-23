import glob, os, argparse

def main(args):
    paths = list(glob.glob(args.target_dir + '/*.png')) + list(glob.glob(args.target_dir + '/*.jpg'))
    for path in paths:
        header = '.'.join(path.split('.')[:-1])
        # print(os.path.exists(header + '.xml'))
        if not os.path.exists(header + '.xml'):
            os.remove(path)

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('target_dir', type=str)
      args = parser.parse_args()
      main(args)
