import numpy as np
import math


# Function in 2D
def f2d(x, y):
    a, b = 2, 19
    return pow((a - x), 2) + b * pow((y - pow(x, 2)), 2)


# Function in 3D
def f3d(x, y, z):
    a, b = 1, 18
    return (pow(a - x, 2) + b * (pow(y - pow(x, 2), 2))) + (
        pow(a - y, 2) + b * (pow(z - pow(y, 2), 2))
    )


# Sort simplex points in ascending manner
def sort(simplex, f):
    # setting a helper variable to shape
    colShape = simplex.shape[1]+1
    # creating a new column for the matrix to store values in order to use numpy sorting function
    # reshape = changes the row shape to the column shape of
    # inner function = create a numpy array based on list comprehension that creates a list of values for given x, y, z
    values = np.reshape((np.array([f(*simplex[i]) for i in range(len(simplex))])), (colShape, 1))
    # adding the column to the simplex
    with_values = np.hstack((simplex, values))
    # sorting the points from the smallest
    sorted_simplex = with_values[with_values[:, simplex.shape[1]].argsort()]
    # cut the column with values, return only a sorted simplex
    return np.delete(sorted_simplex, -1, 1)


def nelder_mead(simplex, f, e):
    alpha, gamma, beta, delta = 1, 2, 0.5, 0.5
    dim = simplex.shape[1]
    n = len(simplex)
    while True:
        # Step 1: Sort points
        simplex = sort(simplex, f)
        # Step 1.a: Find centroid xo
        # delete to not take into account the last point (we take the best ones)
        xo = np.sum(simplex[:-1], axis=0)/dim
        # Step 2: Reflection xR
        xR = (1 + alpha) * xo - alpha * simplex[-1]
        # Step 3: Check if val of xR is smaller than value of x1
        if f(*xR) < f(*simplex[0]):
            # Step 3.a introducing expansion and comparing value to x1
            # Expansion:
            xE = gamma * xR + (1 - gamma) * xo
            if f(*xE) < f(*simplex[0]):
                simplex[-1] = xE
            else:
                simplex[-1] = xR
        else:
            # Step 4: check and replace for xR if smaller
            if f(*simplex[0]) <= f(*xR) <= f(*simplex[-2]):
                simplex[-1] = xR
            else:
                # Step 5: if condition is met, introduce contraction
                if f(*simplex[-2]) < f(*xR) < f(*simplex[-1]):
                    # Contraction
                    xC = beta * simplex[-1] + (1 - beta) * xo
                    # Step 5.a: check if smaller than last point if so, change for xC otherwise reduce
                    if f(*xC) < f(*simplex[-1]):
                        simplex[-1] = xC
                    else:
                        # Reduction / Shrinking, therefore xS
                        xS = np.array([delta * (simplex[i] + simplex[0]) for i in range(n)])
                        simplex = xS
                else:
                    # Step 6: if value of xR >= value of xn+1 then introduce xC
                    if f(*xR) >= f(*simplex[-1]):
                        # Contraction xC
                        xC = beta * simplex[-1] + (1 - beta) * xo
                        # Step 6.a: if value of xC is smaller, change to xC, otherwise reduce
                        if f(*xC) < f(*simplex[-1]):
                            simplex[-1] = xC
                        else:
                            # Reduction
                            xS = np.array([delta * (simplex[i] + simplex[0]) for i in range(n)])
                            simplex = xS
        # Step 7: stop criteria
        crit = math.sqrt(
            (1 / n) * sum([pow(f(*simplex[i]) - f(*xo), 2) for i in range(n)])
        )
        if crit < e:
            print(f'Minimum of the function has value: {crit}')
            print("Output simplex [x y] or [x y z] is as follows:")
            print(np.matrix(simplex))
            return


# start of the program
if __name__ == '__main__':
    e = 10e-6
    # initial simplex for 2D and 3D
    simplex_2d = np.array([[7.0, 2.0], [8.0, 2.0], [12., -6.]])
    simplex_3d = np.array([[2., 1., 0.], [4., 1., 0.], [0., 5., 1.], [5., 7., 2.]])
    print("=================================== 2D Version: ")
    nelder_mead(simplex_2d, f2d, e)
    print("=================================== 3D Version: ")
    nelder_mead(simplex_3d, f3d, e)