# Calculates ones when the parameter set gives a non complex eigenvalues and zeroes when it is a complex value
# Result to be imported in Plot_Eigenvalues

import numpy as np

############################################# f1 = k1 & k2 ###########################################################
from Parameter_set import k3, k4, k5, k6, k7

k1 = np.arange(1, 100)
k2 = np.arange(1, 100)
k1k1, k2k2 = np.meshgrid(k1, k2, sparse=True)

p = ((0.1 / 2) ** 2 + k5 / k7) ** 0.5

Mig1 = p - 0.1 / 2

a = 1
b = k4 + k2k2 + k7 + k1k1 + k6

c = k2k2 * k4 + k2k2 * k6 + k1k1 * k7 + k4 * k7 + k6 * k7 + k1k1 * k4 + k1k1 * k6 + k4 * k6 + (k5 * k6) / ((Mig1 + 0.1) ** 2)
d = k2k2 * k4 * k6 + k1k1 * k4 * k7 + k1k1 * k6 * k7 + k4 * k6 * k7 + k1k1 * k4 * k6 + (k1k1 * k5 * k6) / a + (k4 * k5 * k6) / (
        (Mig1 + 0.1) ** 2)
e = (k1k1 * k4 * k5 * k6) / ((Mig1 + 0.1) ** 2) + k1k1 * k4 * k6 * k7


p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)


x1 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 - p6) / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 - p6) / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 + p6) / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 + p6) / 2

f1=x4/x4

######################################### f2 = k4 & k5 ###############################################################

from Parameter_set import k1, k2, k3, k6, k7

k4 = np.arange(1, 100)
k5 = np.arange(1, 100)
k4k4, k5k5 = np.meshgrid(k4, k5, sparse=True)

p = ((0.1 / 2) ** 2 + k5k5 / k7) ** 0.5

Mig1 = p - 0.1 / 2

a = 1
b = k4k4 + k2 + k7 + k1 + k6

c = k2 * k4k4 + k2 * k6 + k1 * k7 + k4k4 * k7 + k6 * k7 + k1 * k4k4 + k1 * k6 + k4k4 * k6 + (k5k5 * k6) / ((Mig1 + 0.1) ** 2)
d = k2 * k4k4 * k6 + k1 * k4k4 * k7 + k1 * k6 * k7 + k4k4 * k6 * k7 + k1 * k4k4 * k6 + (k1 * k5k5 * k6) / a + (k4k4 * k5k5 * k6) / (
        (Mig1 + 0.1) ** 2)
e = (k1 * k4k4 * k5k5 * k6) / ((Mig1 + 0.1) ** 2) + k1 * k4k4 * k6 * k7


p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)


x1 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 - p6) / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 - p6) / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 + p6) / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 + p6) / 2

f2=x4/x4
##################################################### f3 = k5 & k7 ####################################################
from Parameter_set import k1, k2, k3, k4, k6

k5 = np.arange(1, 100)
k7 = np.arange(1, 100)
k5k5, k7k7 = np.meshgrid(k5, k7, sparse=True)

p = ((0.1 / 2) ** 2 + k5k5 / k7k7) ** 0.5

Mig1 = p - 0.1 / 2

a = 1
b = k4 + k2 + k7k7 + k1 + k6

c = k2 * k4 + k2 * k6 + k1 * k7k7 + k4 * k7k7 + k6 * k7k7 + k1 * k4 + k1 * k6 + k4 * k6 + (k5k5 * k6) / ((Mig1 + 0.1) ** 2)
d = k2 * k4 * k6 + k1 * k4 * k7k7 + k1 * k6 * k7k7 + k4 * k6 * k7k7 + k1 * k4 * k6 + (k1 * k5k5 * k6) / a + (k4 * k5k5 * k6) / (
        (Mig1 + 0.1) ** 2)
e = (k1 * k4 * k5k5 * k6) / ((Mig1 + 0.1) ** 2) + k1 * k4 * k6 * k7k7


p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)


x1 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 - p6) / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 - p6) / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 + p6) / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 + p6) / 2

f3=x4/x4

############################################ f4 = k1 & k6 ##############################################################
from Parameter_set import k2, k3, k4, k5, k7

k1 = np.arange(1, 100)
k6 = np.arange(1, 100)
k1k1, k6k6 = np.meshgrid(k1, k6, sparse=True)

p = ((0.1 / 2) ** 2 + k5 / k7) ** 0.5

Mig1 = p - 0.1 / 2

a = 1
b = k4 + k2 + k7 + k1k1 + k6k6

c = k2 * k4 + k2 * k6k6 + k1k1 * k7 + k4 * k7 + k6k6 * k7 + k1k1 * k4 + k1k1 * k6k6 + k4 * k6k6 + (k5 * k6k6) / ((Mig1 + 0.1) ** 2)
d = k2 * k4 * k6k6 + k1k1 * k4 * k7 + k1k1 * k6k6 * k7 + k4 * k6k6 * k7 + k1k1 * k4 * k6k6 + (k1k1 * k5 * k6k6) / a + (k4 * k5 * k6k6) / (
        (Mig1 + 0.1) ** 2)
e = (k1k1 * k4 * k5 * k6k6) / ((Mig1 + 0.1) ** 2) + k1k1 * k4 * k6k6 * k7


p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)


x1 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 - p6) / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 - p6) / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 + p6) / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 + p6) / 2

f4=x4/x4

############################################ f5 = k6 & k7 #############################################################
from Parameter_set import k1, k2, k3, k4, k5

k6 = np.arange(1, 100)
k7 = np.arange(1, 100)
k6k6, k7k7 = np.meshgrid(k6, k7, sparse=True)

p = ((0.1 / 2) ** 2 + k5 / k7k7) ** 0.5

Mig1 = p - 0.1 / 2

a = 1
b = k4 + k2 + k7k7 + k1 + k6k6

c = k2 * k4 + k2 * k6k6 + k1 * k7k7 + k4 * k7k7 + k6k6 * k7k7 + k1 * k4 + k1 * k6k6 + k4 * k6k6 + (k5 * k6k6) / ((Mig1 + 0.1) ** 2)
d = k2 * k4 * k6k6 + k1 * k4 * k7k7 + k1 * k6k6 * k7k7 + k4 * k6k6 * k7k7 + k1 * k4 * k6k6 + (k1 * k5 * k6k6) / a + (k4 * k5 * k6k6) / (
        (Mig1 + 0.1) ** 2)
e = (k1 * k4 * k5 * k6k6) / ((Mig1 + 0.1) ** 2) + k1 * k4 * k6k6 * k7k7


p1 = 2 * c ** 3 - 9 * b * c * d + 27 * a * d ** 2 + 27 * b ** 2 * e - 72 * a * c * e
p2 = p1 + ((-4) * (c ** 2 - 3 * b * d + 12 * a * e) ** 3 + p1 ** 2) ** 0.5
p3 = (c ** 2 - 3 * b * d + 12 * a * e) / (3 * a * (p2 / 2) ** (1 / 3)) + ((p2 / 2) ** (1 / 3)) / (3 * a)
p4 = ((b ** 2) / (4 * a ** 2) - (2 * c) / (3 * a) + p3) ** 0.5
p5 = b ** 2 / (2 * a ** 2) - 4 * c / (3 * a) - p3
p6 = (b ** 3 / a ** 3 + 4 * b * c / a ** 2 - 8 * d / a) / (4 * p4)


x1 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 - p6) / 2
x2 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 - p6) / 2
x3 = -b / (2 * a ** 2) - p4 / 2 - np.sqrt(p5 + p6) / 2
x4 = -b / (2 * a ** 2) - p4 / 2 + np.sqrt(p5 + p6) / 2

f5=x4/x4
