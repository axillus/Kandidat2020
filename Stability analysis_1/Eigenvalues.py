# Calculates the eigenvalues of the Jacobian matrix
# The eigenvalues are referred to as x
# Equation of the form ax^4 + bx^3 + cx^2 + dx + e = 0
# Derived from the characteristic equation Det(J - xI) = 0 where J is the Jacobian matrix.
from Parameter_set import k1, k2, k3, k4, k5, k6, k7

p = ((0.1 / 2) ** 2 + k5 / k7) ** 0.5

Mig1 = p - 0.1 / 2
Mig1Phos = k2 / k1 * (p - 0.1 / 2)
Suc2 = (k3 / k4) * (1 / (p + 0.1 / 2))
X = (k5 / k6) * (1 / (p + 0.1 / 2))

a = 1
b = k4 + k2 + k7 + k1 + k6
c = k2 * k4 + k2 * k6 + k1 * k7 + k4 * k7 + k6 * k7 + k1 * k4 + k1 * k6 + k4 * k6 + (k5 * k6) / ((Mig1 + 0.1) ** 2)
d = k2 * k4 * k6 + k1 * k4 * k7 + k1 * k6 * k7 + k4 * k6 * k7 + k1 * k4 * k6 + (k1 * k5 * k6) / a + (k4 * k5 * k6) / (
            (Mig1 + 0.1) ** 2)
e = (k1 * k4 * k5 * k6) / ((Mig1 + 0.1) ** 2) + k1 * k4 * k6 * k7

p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)

x1 = -b / (2 * a ** 2) - p4 / 2 - (p5 - p6) ** 0.5 / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + (p5 - p6) ** 0.5 / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - (p5 + p6) ** 0.5 / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + (p5 + p6) ** 0.5 / 2


d1 = a
d2 = b*c - d
d3 = b*c*d - b**2*e - d**2
d4 = e

print(x1, x2, x3, x4)

