import math

while True:
    Ut3a = float(input('Ut3a='))
    Ut3b = float(input('Ut3b='))
    Ut1a = float(input('Ut1a='))
    Ut1b = float(input('Ut1b='))
    Ut3 = 0.5 * (Ut3a**2 + Ut3b **2) ** 0.5
    Ut1 = 0.5 * (Ut1a**2 + Ut1b **2) ** 0.5
    tao = abs(Ut3 - Ut1)
    Utao = 0.5 * (Ut3**2 + Ut1 **2) ** 0.5
    k = float(input('k='))
    ma = float(input('ma='))
    mb = float(input('mb='))
    Uka = k * Utao / tao * 4 * math.pi ** 2 * abs(ma-mb)
    print(Uka)
