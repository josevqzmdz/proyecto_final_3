import numpy as np
class IMDB_examples:

    def __init__(self):
        print("")
    # 98
    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1
        return results

    #x_train = vectorize_sequences(train_data)