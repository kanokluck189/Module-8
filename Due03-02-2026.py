from math import sqrt

# Function to check if a point is inside the unit circle
def inside_circle(x, y):
    return x*x + y*y <= 1


def approximate_area(square_size):
    # Bounding box for unit circle is [-1, 1] in both axes
    start = -1
    end = 1
    count_inside = 0
    total = 0
    
    x = start
    while x < end:
        y = start
        while y < end:
            # Check center of the square
            cx = x + square_size / 2
            cy = y + square_size / 2
            
            if inside_circle(cx, cy):
                count_inside += 1
            
            total += 1
            y += square_size
        x += square_size
    
    area_estimate = count_inside * (square_size ** 2)
    return area_estimate


# Try different square sizes
sizes = [0.5, 0.2, 0.1, 0.05]

results = {}
for s in sizes:
    area = approximate_area(s)
    results[s] = area
    print(f"Square size: {s} -> Approximated area: {area}")

print("\nActual area of unit circle (pi):", 3.141592653589793)
