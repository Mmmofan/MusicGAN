import argparse

def process():
    print(args.list)

def main(args):
    process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', default='afe', type=str)
    args = parser.parse_args()
    main(args)