from random import shuffle

#reference: https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/

"""
Creates shingles de comprimento k de um texto.
Parameters:
    text (str): texto que vai passar pelo shingle.
    k (int): de quanto em qunto vai ser dividido o texto.
Returns:
    set: set de shingles.
"""
def shingling(text, k):

    n = len(text)
    shingles = []

    for i in range(n - k + 1):
        
        shingle = text[i:i + k]
        shingles.append(shingle)
    # importante notar que o set do python da o resultado em ordem aleatoria, ou seja, o singles de um texto pode ser diferente dependendo da execução
    return set(shingles)

"""
Gera assinaturas MinHash para uma lista dada de vetores esparsos (de cada frase), uma lista de vocabulário (junção de todas as frases) e o número de bits das assinaturas desejadas.

Parâmetros:
    singles_sparse (list): Uma lista de vetores esparsos.
    vocab (list): Uma lista de elementos únicos no vocabulário.
    bits (int): O número de bits a ser usado para a assinatura MinHash.

Retorna:
    list: Uma lista de assinaturas MinHash, onde cada assinatura é uma lista de índices.
    
"""
def minhash(singles_sparse, vocab, bits):
    # o bits vai definir o tamanho do minhash resultante
    minhash_functions = []
    
    for i in range(bits):
        # seu funcionamento vai se dar com numeros de 1 a len(vocab)+1, e isso será feito de maneira aleatoria
        minhash = [ i for i in range(1,len(vocab)+1)]
        shuffle(minhash)
        minhash_functions.append(minhash)

    signatures = []
    # cada singles_sparse vai possuir uma hash
    for hash in singles_sparse:
        signature = []
        # isso vai ser feito com base em cada hashFunc das funções geradas acima
        for hashFunc in minhash_functions:
            # retorna valor onde o elemento se encontra no vetor de minhash e vê se essa possição no sparse vector é 1
            # uma vez preenchida a assinatura, vamos para o proximo
            for i in range(1, len(vocab)+1):
                idx = hashFunc.index(i)
                if hash[idx] == 1:
                    signature.append(idx)
                    break
        signatures.append(signature)
        
    return signatures

def jaccard_similarity(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

def lsh(text_list, k_shingling, k_hash):
   
    # o vocabulario vai ser o set de shingles de todos os textos que levaremos em conta para busca
    vocabulary = set()
    shingles_list = []
    for text in text_list:
        shingles = shingling(text, k_shingling)
        shingles_list.append(shingles)
        vocabulary = vocabulary.union(shingles)

    # cada texto vai ter um single que reflete sua posicao no vocabulario
    sparse_vectors = []
    for text in text_list:
        shingles = shingles_list[text_list.index(text)]
        # ve em que posicao cada shingle se encontra no vocabulario
        vector = [1 if shingle in shingles else 0 for shingle in vocabulary]
        sparse_vectors.append(vector)

    signatures = minhash(sparse_vectors, vocabulary, k_hash)
    for i in range(len(signatures)):
        for j in range(i+1, len(signatures)):
            similaridade = jaccard_similarity(set(signatures[i]), set(signatures[j]))
            print(f'Similaridade: {similaridade}')
            # if similaridade > 0.5:
            #     print(text_list[i], " | ", text_list[j])
    return signatures

# def main():
#     text_list = ["The quick brown fox jumps over the lazy dog", 
#                 "The quick brown fox jumps over the lazy dog", 
#                 "The brown fox jumps over the lazy dog",
#                 "Sweet dreams are made of this",
#                 "Are they made of this dream?",
#                 "Bitter dreams are made of this",
#             ]
#     k = 3
#     lsh(text_list, k)
 
# main()