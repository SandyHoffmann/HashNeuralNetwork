import numpy as np

# 1 data pixel = 32 bits

#array aleatorio 31 bits
#Ex: [0 1 0 0 1 1 1 1 1 0 0 1 1 1 0 1 0 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0]
T = 50

def funcao_hash(x, Q):
    if (0 <= x < Q):
        return x/Q
    elif(Q <= x < 0.5):
        return (x - Q)/(0.5 - Q)
    elif(0.5 <= x < (1-Q)):
        return (1-Q - x)/(0.5 - Q)
    elif((1-Q) <= x <= 1):
        return ((1 - x)/Q)
    
# deixando numero binário entre 0 e 1 (por conta da função hash)
def binary_to_decimal(binary_string):
    decimal = 0
    for digit in binary_string:
        decimal = decimal * 2 + int(digit)
    return decimal / (2 ** len(binary_string))

# convertendo decimal para binário de 32 bits
def decimal_to_binary(decimal_array):
    binary_array = []
    for decimal in decimal_array:
        binary = bin(int(decimal * (2**32)))[2:]  # Convert to binary, remove '0b' prefix
        binary = '0' * (32 - len(binary)) + binary  # Pad with zeros to ensure 32 bits
        binary_array.append(binary)
    return binary_array

def string_to_binary_array(string, max_size=128):
    # Encode the string into bytes
    bytes_string = string.encode('utf-8')
    
    # Convert each byte to binary representation and concatenate
    binary_array = []
    for byte in bytes_string:
        binary_representation = bin(byte)[2:].zfill(8)  # Convert byte to binary string and pad with zeros
        binary_array.extend(map(int, binary_representation))
    
    # Pad with zeros if necessary to ensure max_size bits
    if len(binary_array) < max_size:
        binary_array = [0] * (max_size - len(binary_array)) + binary_array
    
    return binary_array[:max_size]
def key_gen(K):
    # K key = 128 bits
    # K key = k0, k1, ... , k127
    # K key dividida em 4 subkeys k0, k1, k2, k3, de modo que:
    # k0 = k0k1...k31, k1 = k32k33...k63, k2 = k64k65...k95, k3 = k96k97...k127
    # K max size = 151 data pixels = 4.832 bits = (total de dados em todos os parametros de saida)

    # X0(k) = f**(T+k)(k0, k1)
    # X1(k) = f**(T+k)(k2, k3)
    # Ks(k) = (X0(k) + X1(k))mod 1, onde ks(k) (k = 0, 1, .., 150)

    # transformar string em array de bits de forma que:
    # W0 = 1x32
    # B0 = 8x1
    # W1 = 8x8 
    # B1 = 8x1
    # W2 = 4x8
    # B2 = 4x1
    # Q0 = float
    # Q1 = float
    # Q2 = float
    margem_erro = 1/1000

    x0 = 0
    x1 = 0
    ks = 0

    k0 = ''.join(map(str, K[0:32])) 
    k1 = ''.join(map(str, K[32:64]))
    k2 = ''.join(map(str, K[64:96]))
    k3 = ''.join(map(str, K[96:128]))

    resultados = np.array([])
    # cada data-pixel tem 32 bits

    xk0 = binary_to_decimal(k0)
    xk1 = binary_to_decimal(k1)
    xk2 = binary_to_decimal(k2)
    xk3 = binary_to_decimal(k3)


    for i in range(0, 152):

        for j in range(0, T):
            x0 = funcao_hash(xk0, (xk1/2)-margem_erro)
            xk0 = x0
            x1 = funcao_hash(xk2, (xk3/2)-margem_erro)
            xk2 = x1
            ks_mod = (x0 + x1) % 1
            if 0 <= ks_mod < 1:
                ks = ks_mod
            elif 1 <= ks_mod < 2:
                ks = ks_mod - 1
            # deveria salvar numero em 32 bits 
        resultados = np.append(resultados, ks)
        
    W0 = resultados[:32].reshape(1, 32)
    B0 = resultados[32:40].reshape(8, 1)
    W1 = resultados[40:104].reshape(8, 8)
    B1 = resultados[104:112].reshape(8, 1)
    W2 = resultados[112:144].reshape(4, 8)
    B2 = resultados[144:148].reshape(4, 1)
    Q0 = min(resultados[149], 0.5-margem_erro)  
    Q1 = min(resultados[150], 0.5-margem_erro)
    Q2 = min(resultados[151], 0.5-margem_erro)


    return W0, B0, W1, B1, W2, B2, Q0, Q1, Q2

key = "0123456789abcdef"
binary_array = string_to_binary_array(key)

np.set_printoptions(precision=3)
array = np.array(binary_array)
W0, B0, W1, B1, W2, B2, Q0, Q1, Q2 = key_gen(array)

#frase com 1024 bits
stringFrase = "Cellular neural networks (CNN) chatic secure communication is a new secure communication scheme based on chaotic synchronization"
print("key: ", key + " " + str(len(key)) + " ou " + str(len(key) * 8) + " bits")
print("stringFrase: ", stringFrase + " " + str(len(stringFrase)) + " ou " + str(len(stringFrase) * 8) + " bits")


# frase padrão = 1024 bits

stringFrase_binary = string_to_binary_array(stringFrase, 2**10)
stringFrase_decimal = []

# iterando sobre cada P, para que P0, P1, .., P31 tenham 32 bits
for i in range(0, 32):
    stringFrase_x = stringFrase_binary[i*32:(i+1)*32]
    stringFrase_decimal.append(binary_to_decimal(stringFrase_x))


# Cada data-pixel tem 32 bits
# Minha string frase se dividira em P = [P0, P1, ..., P31]
# O primeiro neuronio C0 receberá como entrada P0, P1, P2 e P3. E assim, de 4 em 4 irá ate C7

def forward_propagation(W0, B0, W1, B1, W2, B2, Q0, Q1, Q2, P):
    # Camada C = 1x8
    C = np.zeros(8)

    for i in range(0, 8):
        # Funcao é aplicada T vezes, onde T>=50
        Ci = 0
        w0i = W0[0][i*4:(i+1)*4]
        # ! talvez melhor juntar 4 bytes em um float de uma vez ao inves da soma em si
        sumW0i = np.sum(w0i * P[i*4:(i+1)*4] + B0[i])
        sumBin = decimal_to_binary([sumW0i])
        sumW0i = binary_to_decimal(sumBin[0])
        for j in (0,T):
            Ci = funcao_hash(min(sumW0i,1), Q0)
            sumW0i = Ci
        C[i] = Ci

    # Camada D = 1x8
    D = np.zeros(8)

    for j in range(0, 8):
        # Utiliza-se o peso W1 e B1
        w1j = 0
        for i in range(0, 8):
            w1j += W1[j][i] * C[i] + B1[j][0]
        
        w1jSum = decimal_to_binary([w1j])
        w1j = binary_to_decimal(w1jSum[0])

        D[j] = funcao_hash(min(w1j, 1), Q1)
    
    # Camada H = 1x4
    H = np.zeros(4)
    for j in range(0, 4):
        # Utiliza-se o peso W2 e B2
        w2j = 0
        for i in range(0, 8):
            w2j += W2[j][i] * D[i] + B2[j][0]
        w2jSum = decimal_to_binary([w2j])
        w2j = binary_to_decimal(w2jSum[0])

        for t in (0,T):
             w2j = funcao_hash(min(w2j,1), Q2)
       
        H[j] = w2j

    print("C: ",C)
    print("D: ",D)
    print("H: ",H)

    hash_h = decimal_to_binary(H)
    hex_strings = [hex(int(binary, 2))[2:].upper() for binary in hash_h]
    print("stringFrase_hex: ", hex_strings)

forward_propagation(W0, B0, W1, B1, W2, B2, Q0, Q1, Q2, stringFrase_decimal)

# print(stringFrase_binary)

# print("W0:\n", np.array2string(W0, precision=2, separator=', '))
# print("B0:\n", np.array2string(B0, precision=2, separator=', '))
# print("W1:\n", np.array2string(W1, precision=2, separator=', '))
# print("B1:\n", np.array2string(B1, precision=2, separator=', '))
# print("W2:\n", np.array2string(W2, precision=2, separator=', '))
# print("B2:\n", np.array2string(B2, precision=2, separator=', '))
# print("Q0: ", Q0)
# print("Q1: ", Q1)
# print("Q2: ", Q2)
# print("W0[0] = ", W0[0])
# binary_representations = [decimal_to_binary(decimal) for decimal in W0]
# print(binary_representations)
