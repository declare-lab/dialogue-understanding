import numpy as np

def load_pretrained_glove(path):
    f = open(path, encoding='utf-8')
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading GloVe model.")
    return glv_vector

def pretrained_matrix(path, id_to_token):
    glv_vector = load_pretrained_glove(path)
    dim = len(glv_vector[list(glv_vector.keys())[0]])
    matrix = np.zeros((len(id_to_token), dim))

    for j in range(len(id_to_token)):
        try:
            matrix[j] = glv_vector[id_to_token[j]]
        except KeyError:
            continue

    return matrix
