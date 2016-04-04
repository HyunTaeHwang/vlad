import os
import scipy.io

devkit_path = os.path.join("data", 'ILSVRC2014/ILSVRC2014_devkit')
print devkit_path
data_path = os.path.join(devkit_path, 'data')
 
def main():
    synsets = scipy.io.loadmat(data_path + "/meta_det.mat")
    for i in xrange(200):
        print synsets['synsets'][0][i][0][0], synsets['synsets'][0][i][1][0], synsets['synsets'][0][i][2][0]
 
if __name__ == "__main__":
    main()
    