# coding=utf-8
import sys, os


def rmCurDir():
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	sys.path = [p for p in sys.path if os.path.abspath(p) != SCRIPT_DIR]


def getRootDir():
	rmCurDir()
	if getattr(sys, 'frozen', False):
		root = os.path.dirname(os.path.abspath(sys.executable))
	else:
		root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
	return root


sys.path.append(getRootDir())


def test_get_datasets_dir():
	from anomalib.utils.path import get_datasets_dir
	get_datasets_dir()


if __name__ == '__main__':
	# print(getRootDir())
	test_get_datasets_dir()
