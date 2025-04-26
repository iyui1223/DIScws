import numpy as np

class DataCompressor:
    """
    Handles compression and decompression of trajectory data.
    """
    PERCENTILE_Q = 99
    MAX_VALUE = 999
    TARGET = {0: "prey", 1: "predator"}

    @classmethod
    def compress(cls, trajectory, qtile_value):
        """Compresses float32 data to integers (0-999)."""
        return np.clip(np.round(trajectory / qtile_value * cls.MAX_VALUE), 0, cls.MAX_VALUE).astype(np.int16)

    @classmethod
    def decompress(cls, trajectory, qtile_value):
        """Decompresses integers (0-999) back to float32 values."""
        return trajectory * qtile_value / cls.MAX_VALUE

    @classmethod
    def save_qtile(cls, i, qtile_value):
        """Saves the percentile threshold for later use."""
        np.save(f"{cls.TARGET[i]}{cls.PERCENTILE_Q}value.npy", qtile_value)

    @classmethod
    def load_qtile(cls, i):
        """Loads the saved percentile threshold."""
        return np.load(f"{cls.TARGET[i]}{cls.PERCENTILE_Q}value.npy")


class DataConverter:
    """
    Handles conversion between different data representations.
    """
    @staticmethod
    def univariate(intarray):
        """Converts a multi-variable time series into a univariate string format."""
        from einops import rearrange
        systems, tsteps, types = intarray.shape
        uni = rearrange(intarray, 'systems tsteps type -> tsteps type systems')
        return [[";".join([",".join(map(str, uni[i, :, j])) for i in range(tsteps)])] for j in range(systems)]

    @staticmethod
    def bivariate(inputlist):
        """Converts a univariate string representation back into structured NumPy arrays."""
        systems = len(inputlist)
        tsteps = len(inputlist[0].split(";"))
        arr = np.zeros((systems, tsteps, 2), dtype=np.int16)
        for i, inputstr in enumerate(inputlist):
            temp = inputstr.lstrip(";").split(";")
            for j, entry in enumerate(temp):
                arr[i, j, :] = list(map(int, entry.split(",")))
        return arr


class TransformerProcessor:
    """
    Handles tokenization and de-tokenization using a transformer model.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, sequences):
        """Tokenizes input sequences for the transformer model."""
        return [self.tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].tolist()[0] for seq in sequences]


class PrePostProcessor:
    """
    Handles the full pipeline of preprocessing and postprocessing.
    """
    def __init__(self, transformer_model="Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = TransformerProcessor(transformer_model)

    def preprocess(self, time, trajectories):
        """Processes raw data into tokenized input for a transformer model."""
        compressed = np.zeros_like(trajectories, dtype=np.int16)
        
        for i in DataCompressor.TARGET.keys():
            qtile_value = np.percentile(trajectories[:, :, i].flatten(), DataCompressor.PERCENTILE_Q)
            compressed[:, :, i] = DataCompressor.compress(trajectories[:, :, i], qtile_value)
            DataCompressor.save_qtile(i, qtile_value)
        
        univariated = DataConverter.univariate(compressed)
        return self.tokenizer.tokenize(univariated)


    def univariate(self, time, trajectories):
        """Processes raw data into tokenized input for a transformer model."""
        compressed = np.zeros_like(trajectories, dtype=np.int16)
        
        for i in DataCompressor.TARGET.keys():
            qtile_value = np.percentile(trajectories[:, :, i].flatten(), DataCompressor.PERCENTILE_Q)
            compressed[:, :, i] = DataCompressor.compress(trajectories[:, :, i], qtile_value)
            DataCompressor.save_qtile(i, qtile_value)
        
        return DataConverter.univariate(compressed)
    

    def postprocess(self, output):
        """Converts model output back into structured numerical data."""
        try:
            bivariated = DataConverter.bivariate([output])
            decompressed = np.zeros_like(bivariated, dtype=np.float32)
            
            for i in DataCompressor.TARGET.keys():
                qtile_value = DataCompressor.load_qtile(i)
                decompressed[:, :, i] = DataCompressor.decompress(bivariated[:, :, i], qtile_value)
            
            return decompressed

        except: # when returned prediction is not in expected format
            return output


def read_hdf5(infile):
    """Reads an HDF5 file containing Lotka-Volterra model output."""
    import h5py
    with h5py.File(infile, 'r') as f:
        time = f["time"][:]
        trajectories = f["trajectories"][:]
    return time, trajectories


def load_and_preprocess(infile):
    time, trajectories = read_hdf5(infile)
    processor = PrePostProcessor()
    univariated = processor.univariate(time, trajectories)
    split_index = int(len(univariated) * 0.7)

    return univariated[:split_index], univariated[split_index:]


if __name__ == "__main__":
    """ Test script """
    infile = "./lotka_volterra_data.h5"
    time, trajectories = read_hdf5(infile)

    processor = PrePostProcessor()

    print("input")
    print(trajectories[:2, :, :])

    # tokenized = processor.preprocess(time, trajectories[:2, :, :])

    # prediction = processor.postprocess(";699,739;710,644;698,777;692,696;655,806;612,763;595,585;552,921;521,792;487,904;465,748;423,825;392,840;363,762;342,715;321,564;305,482;289,759;278,715;266,689;262,569;260,432;257,374;248,342;242,999;189,757;174,548;183,396;210,292;256,223;322,181;409,158;517,150;643,157;775,182;896,230;984,312;999,438;962,612;829,806;650,954;480,986;359,901;291,753;263,602;262,474;282,377;321,309;378,266;451,242")  
    # print(prediction)
