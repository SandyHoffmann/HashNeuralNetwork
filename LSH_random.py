import numpy as np

def lsh_random_projection(list_numbers, nbits, d):
    norm_plan = np.random.randn(nbits, d)
    norms = np.linalg.norm(norm_plan, axis=1)
    norm_plan = norm_plan / norms[:, np.newaxis]
    print(f'norm_plan.shape: {norm_plan.shape}')
    transform_plan = np.dot(list_numbers, norm_plan)
    # transform_plan = np.array(list_numbers)
    print(f'transform_plan.shape: {transform_plan.shape}')
    print(f'transform_plan.shape: {transform_plan.shape}')
    print(f'transform_plan.shape: {transform_plan.shape}')

    binary_plan = (transform_plan > 0).astype(int)
    print(f'binary_plan.shape: {binary_plan.shape}')

    # print(binary_plan[0])
    lsh_bucket = lsh_buckets(binary_plan, d)
    hamming_distance(lsh_bucket)
    print(lsh_bucket)
    return lsh_bucket

def lsh_buckets(list_binary, k):

    dict_buckets = {}
    for i in range(len(list_binary)):
        print(list_binary[i])
        key = ''.join(list_binary[i][0].astype(str))
        if key in dict_buckets:
            dict_buckets[key].append(i)
        else:
            dict_buckets[key] = [i]

    with open('lsh_bucket.txt', 'w') as f:
        for key, value in dict_buckets.items():
            f.write(f"{key}: {value}\n")

    
    return dict_buckets

def hamming_distance(lsh_bucket):
    keys = list(lsh_bucket.keys())
    relacao = {}
    indices_imagem = {}
    for i in range(len(keys)):
        indices_imagem[f'Figura {i}'] = {}
        for j in range(len(keys)):
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(keys[i], keys[j]))
            if relacao.get(hamming_distance):
                relacao[hamming_distance].append((i, j))
            else:
                relacao[hamming_distance] = [(i, j)]

            if indices_imagem[f'Figura {i}'].get(hamming_distance):
                indices_imagem[f'Figura {i}'][hamming_distance].append(j)
            else:
                indices_imagem[f'Figura {i}'][hamming_distance] = [j]
        indices_imagem[f'Figura {i}'] = dict(sorted(indices_imagem[f'Figura {i}'].items(), key=lambda x: x[0]))
    relacao = dict(sorted(relacao.items(), key=lambda x: x[0]))

    # return relacao in a txt
    with open('relacao.txt', 'w') as f:
        for key, value in relacao.items():
            f.write(f"{key}: {value}\n")

    with open('indices_imagem.txt', 'w') as f:
        for key, value in indices_imagem.items():
            f.write(f"{key}: {value}\n")
    # print(indices_imagem)



def main():
    pass