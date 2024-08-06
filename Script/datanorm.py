import pandas as pd
import numpy as np

class PeakProcessor:
    def __init__(self, mass_range = 1000, bins = 100, pkl_path = "/home/code/mol2sepc/data/train.pkl") -> None:
        self.mass_range = mass_range
        self.bins = bins
        self.bin_size = mass_range / bins
        self.pkl_path = pkl_path
        

    def process_peaks(self, peaks, bins, bin_size):
        histogram = np.zeros((bins, 2))
        for peak in peaks:
            mass, intensity = peak
            bin_index = int(mass / bin_size)
            if bin_index < bins:
                histogram[bin_index, 0] = mass
                histogram[bin_index, 1] += intensity

        return histogram

    def get_peaks_feature(self):
        df = pd.read_pickle(self.pkl_path)
        peaks_ndarray = df["peaks"].values
        peaks_list = []
        for peak_n in peaks_ndarray:
            peaks_array = np.frombuffer(peak_n, dtype=np.float64).reshape(-1, 2)
            peaks_list.append(self.process_peaks(peaks_array, self.bins, self.bin_size))

        peaks_feature = np.array(peaks_list)

        # print(len(smiles),len(peaks_feature))
        # print(peaks_list[0:3])
        # print(peaks_feature[0:3])
        return peaks_feature
    

if __name__ == "__main__":
    peakp = PeakProcessor(1000, 200 ,"/home/code/mol2sepc/data/train.pkl")
    np.set_printoptions(threshold=np.inf)
    peaks = peakp.get_peaks_feature()
    print(len(peaks))
    print(peaks[:2])
 
    

