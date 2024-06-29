def media(valores):
    num_val = len(valores)
    media = sum(valores) / num_val
    return media

def desvio_padrao(valores):
    num_val = len(valores)
    average = media(valores)
    variancia = sum([((x - average) ** 2) for x in valores]) / num_val
    desvioPadrao = variancia ** 0.5
    return desvioPadrao

def main():
    valores = []
    num_val = 4
    print("Digite os valores: ")
    [valores.append(float(input())) for _ in range(num_val)]

    var_media = media(valores)
    print("A media dos valores é: ", var_media)

    var_dp = desvio_padrao(valores)
    print("O desvio padrão dos valores é: ", var_dp)

main()