import math

def calcular_intensidad_lluvia(t, T):
    # Primera parte de la ecuación
    primera_parte = 39.3015 * ((t + 20) ** 0.9228)

    # Segunda parte de la ecuación
    segunda_parte = 10.1767 * ((t + 20) ** 0.8764) * (-0.4653 - 0.8407 * math.log(T/(T-1)))

    # Sumamos las dos partes para obtener la intensidad total de la lluvia
    i = primera_parte + segunda_parte

    return i

# Valores iniciales
t = 10
T = 5

for _ in range(10, 60):
    intensidad = calcular_intensidad_lluvia(t, T)
    print(f"La intensidad de lluvia para t = {t} y T = {T} es: {intensidad} mm/min")

    # Incrementamos los valores de t y T
    t = t + 10
    T = T + 5

