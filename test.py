def solve():
    from math import sqrt

    # Given lengths
    AB = 5
    BC = 9
    AC = 10

    # Using the formula for the area of triangle ABC using Heron's formula
    s = (AB + BC + AC) / 2  # semi-perimeter
    area = sqrt(s * (s - AB) * (s - BC) * (s - AC))

    # Circumradius R of triangle ABC
    R = (AB * BC * AC) / (4 * area)

    # Using the power of point D with respect to the circumcircle
    # The power of point D is equal to the product of the lengths of the tangents from D to the circle
    # which is equal to the product of the lengths of the segments AD and DP
    # We can find AD using the formula AD = R * (AB + AC) / BC
    AD = R * (AB + AC) / BC

    # Now we need to find AP
    # By the power of point theorem, AP * PD = AD^2
    # Since P is on the circle, PD = AD - AP
    # Therefore, AP * (AD - AP) = AD^2
    # This leads to the quadratic equation: AP^2 - AD * AP + AD^2 = 0

    # Solving the quadratic equation for AP
    a = 1
    b = -AD
    c = AD**2

    # Discriminant
    D = b**2 - 4 * a * c
    if D < 0:
        return 'No real solution'

    # Roots of the quadratic equation
    AP1 = (-b + sqrt(D)) / (2 * a)
    AP2 = (-b - sqrt(D)) / (2 * a)

    # We take the positive root since lengths are positive
    AP = AP1

    # Express AP as a fraction m/n
    from fractions import Fraction
    fraction = Fraction(AP).limit_denominator()
    m = fraction.numerator
    n = fraction.denominator

    return m + n

# Call the solve function and print the result
result = solve()
print(result)