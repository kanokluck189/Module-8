def f(x):
    return x**4 + 3*x**3 + x**2 - 2*x - 0.5

def bisection(func, a, b, tol=1e-7):
    """Standard bisection to find a root within [a, b]."""
    if func(a) * func(b) >= 0:
        return None
    
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if func(midpoint) == 0:
            return midpoint
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2

def find_all_roots(func, start, end, steps=1000):
    """Heuristic to find multiple roots by splitting the interval."""
    roots = []
    interval_size = (end - start) / steps
    
    for i in range(steps):
        a = start + i * interval_size
        b = a + interval_size
        
        # Check for sign change
        if func(a) * func(b) < 0:
            root = bisection(func, a, b)
            if root is not None:
                roots.append(round(root, 6))
        # Handle the rare case where the step hits the root exactly
        elif func(a) == 0:
            if not roots or abs(roots[-1] - a) > 1e-5:
                roots.append(round(a, 6))
                
    return roots

# Parameters
start_interval = -3
end_interval = 2
resolution = 1000

found_roots = find_all_roots(f, start_interval, end_interval, resolution)

print(f"Roots found in [{start_interval}, {end_interval}]:")
print(found_roots)