# coding=utf-8
import sys, os

def getRootDir():
	if getattr(sys, 'frozen', False):
		root = os.path.dirname(os.path.abspath(sys.executable))
	else:
		root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
	return root
sys.path.append(getRootDir())


# def setCurPath(filename):
# 	currentPath = os.path.dirname(filename)
# 	if currentPath != "":
# 		os.chdir(currentPath)


def test_get_datasets_dir():
	from anomalib.utils.path import get_datasets_dir
	# get_datasets_dir()


if __name__ == '__main__':
	print(getRootDir())
	test_get_datasets_dir()
