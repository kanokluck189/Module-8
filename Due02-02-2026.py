# Function whose root we want to find
def f(x):
    return x**3 - x - 2   # Example function (root is near 1.52)


def bisection(a, b):
    # Check if the interval is valid
    if f(a) * f(b) > 0:
        print("The function must have opposite signs at a and b.")
        return None

    for i in range(100):
        c = (a + b) / 2
        print(f"Iteration {i+1}: center = {c}")

        # Decide which half to keep
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c


# Interval
a = 1
b = 2

root = bisection(a, b)
print("Approximate root:", root)
