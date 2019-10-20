# -*- coding: utf-8 -*-

from numpy import asarray

ENERGY_LEVEL = [100, 113, 110, 85, 105, 102, 86, 63, 81, 101,
                94, 106, 101, 79, 94, 90, 97]


# ==============================================================

# The brute force method to solve first problem
def find_significant_energy_increase_brute(A):
    """
        Return a tuple (i,j) where A[i:j] is
        the most significant energy increase period.
        time complexity = O(n^2)
    """
    inputArray = []
    for i in range(len(A) - 1):
        inputArray.append(A[i + 1] - A[i])
    greatestSum = min(inputArray)
    maxSubArray = (0, 0)
    for i in range(len(inputArray)):
        total = 0
        for j in range(i, len(inputArray)):
            total = total + inputArray[j]
            if total > greatestSum:
                greatestSum = total
                maxSubArray = (i, j)
    x = maxSubArray[1] + 1
    maxSubArray = maxSubArray[:1] + (x,)
    return maxSubArray


# ==============================================================

# The recursive method to solve first problem
def find_significant_energy_increase_recursive(A):
    """
    Return a tuple (i,j) where A[i:j] is
    the most significant energy increase period.
    time complexity = O (n logn)
    """
    inputArray = []
    for i in range(len(A) - 1):
        inputArray.append(A[i + 1] - A[i])
    maxSubArrayAndMaxSum = maxSubArrayRec(inputArray, 0, len(inputArray) - 1)
    maxSubArray = maxSubArrayAndMaxSum[0]
    x = maxSubArray[1] + 1
    maxSubArray = maxSubArray[:1] + (x,)
    return maxSubArray


def maxSubArrayRec(A, left, right):
    if left == right:
        return (left, left), A[left]
    midPoint = (right + left) // 2
    sumL = maxSubArrayRec(A, left, midPoint)
    sumR = maxSubArrayRec(A, midPoint + 1, right)
    sumC = findMaxCrossingSubArray(A, left, right)
    if (sumL[1] >= sumR[1]) and (sumL[1] >= sumC[1]):
        largest = sumL
    elif (sumR[1] >= sumL[1]) and (sumR[1] >= sumC[1]):
        largest = sumR
    else:
        largest = sumC
    return largest


def findMaxCrossingSubArray(A, left, right):
    mid = (right + left) // 2
    greatestSumEndingAtMid = min(A)
    totalSumEndingAtMid = 0
    maxSubArrayEndingAtMid = (mid, mid)
    for i in range(mid, left - 1, -1):
        totalSumEndingAtMid += A[i]
        if totalSumEndingAtMid > greatestSumEndingAtMid:
            greatestSumEndingAtMid = totalSumEndingAtMid
            maxSubArrayEndingAtMid = (i, mid)

    greatestSumStartingAtMidP1 = min(A)
    totalSumStartingAtMidP1 = 0
    maxSubArrayStartingAtMidP1 = (mid + 1, mid + 1)
    for i in range(mid + 1, right + 1):
        totalSumStartingAtMidP1 += A[i]
        if totalSumStartingAtMidP1 > greatestSumStartingAtMidP1:
            greatestSumStartingAtMidP1 = totalSumStartingAtMidP1
            maxSubArrayStartingAtMidP1 = (mid + 1, i)

    maxCrossingSubArray = (maxSubArrayEndingAtMid[0],
                           maxSubArrayStartingAtMidP1[1])
    return maxCrossingSubArray, (greatestSumStartingAtMidP1 +
                                 greatestSumEndingAtMid)


# ==============================================================

# The iterative method to solve first problem
def find_significant_energy_increase_iterative(A):
    """
    Return a tuple (i,j) where A[i:j] is
    the most significant energy increase period.
    time complexity = O(n)
    """
    sumOfMaxSubArray = 0
    maxSubArrayEndingAtI = (0, 0)
    maxSubArraySoFar = (0, 0)
    total = 0
    inputArray = []
    for i in range(len(A) - 1):
        inputArray.append(A[i + 1] - A[i])

    if not (any(i > 0 for i in inputArray)):
        return "Unsolvable using Kadane's since Kadane's requires atleast one positive number in the input array"

    for i in range(len(inputArray)):

        if inputArray[i] > inputArray[i] + total:
            maxSubArrayEndingAtI = (i, i)
            total = inputArray[i]
        else:
            maxSubArrayEndingAtI = maxSubArrayEndingAtI[:1] + (i,)
            total += inputArray[i]

        if sumOfMaxSubArray < total:
            sumOfMaxSubArray = total
            maxSubArraySoFar = maxSubArrayEndingAtI

    x = maxSubArraySoFar[1] + 1
    maxSubArray = maxSubArraySoFar[:1] + (x,)
    return maxSubArray


# ==============================================================

# The Strassen Algorithm to do the matrix multiplication
def square_matrix_multiply_strassens(A, B):
    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2
    """

    A = asarray(A)

    B = asarray(B)

    assert A.shape == B.shape

    assert A.shape == A.T.shape

    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    if len(A) == 1:
        return A[0][0] * B[0][0]

    A00 = A[0:len(A) // 2, 0:len(A) // 2].copy()
    A01 = A[0:len(A) // 2, len(A) // 2:len(A)].copy()
    A10 = A[len(A) // 2:len(A), 0:len(A) // 2].copy()
    A11 = A[len(A) // 2:len(A), len(A) // 2:len(A)].copy()

    B00 = B[0:len(A) // 2, 0:len(B) // 2].copy()
    B01 = B[0:len(A) // 2, len(B) // 2:len(B)].copy()
    B10 = B[len(B) // 2:len(B), 0:len(B) // 2].copy()
    B11 = B[len(B) // 2:len(B), len(B) // 2:len(B)].copy()

    M1 = square_matrix_multiply_strassens(A00 + A11, B00 + B11)
    M2 = square_matrix_multiply_strassens(A10 + A11, B00)
    M3 = square_matrix_multiply_strassens(A00, B01 - B11)
    M4 = square_matrix_multiply_strassens(A11, B10 - B00)
    M5 = square_matrix_multiply_strassens(A00 + A01, B11)
    M6 = square_matrix_multiply_strassens(A10 - A00, B00 + B01)
    M7 = square_matrix_multiply_strassens(A01 - A11, B10 + B11)

    C = [[0 for x in range(len(A))] for y in range(len(A))]
    C = asarray(C)
    C[:(len(A) // 2), :(len(A) // 2)] = M1 + M4 - M5 + M7
    C[:(len(A) // 2):, (len(A) // 2):] = M3 + M5
    C[(len(A) // 2):, :(len(A) // 2)] = M2 + M4
    C[(len(A) // 2):, (len(A) // 2):] = M1 + M3 - M2 + M6
    return C


# ==============================================================

# Calculate the power of a matrix in O(k)
def power_of_matrix_navie(A, k):
    """
    Return A^k.
    time complexity = O(k)
    """
    Temp = A
    for i in range(k - 1):
        Temp = square_matrix_multiply_strassens(A, Temp)

    return Temp


# ==============================================================

# Calculate the power of a matrix in O(log k)
def power_of_matrix_divide_and_conquer(A, k):
    """
    Return A^k.
    time complexity = O(log k)
    """
    if k == 1:
        return A

    if k % 2 == 0:
        return power_of_matrix_divide_and_conquer(
            square_matrix_multiply_strassens(A, A), k // 2
        )

    return square_matrix_multiply_strassens(
        A, power_of_matrix_divide_and_conquer(
            square_matrix_multiply_strassens(A, A), k // 2
        )
    )


# ==============================================================
def test():
    assert (find_significant_energy_increase_brute(ENERGY_LEVEL) ==
            (7, 11))
    assert (find_significant_energy_increase_recursive(ENERGY_LEVEL) ==
            (7, 11))
    assert (find_significant_energy_increase_iterative(ENERGY_LEVEL) ==
            (7, 11))
    assert ((square_matrix_multiply_strassens([[0, 1], [1, 1]],
                                              [[0, 1], [1, 1]]) ==
             asarray([[1, 1], [1, 2]])).all())
    assert ((power_of_matrix_navie([[0, 1], [1, 1]], 3) ==
             asarray([[1, 2], [2, 3]])).all())
    assert ((power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], 3) ==
             asarray([[1, 2], [2, 3]])).all())

    print("Maximum Sub Array Brute Force:")
    print(find_significant_energy_increase_brute(ENERGY_LEVEL))
    print("\n--------------------------------------------------------\n")

    print("Maximum Sub Array Recursive:")
    print(find_significant_energy_increase_recursive(ENERGY_LEVEL))
    print("\n--------------------------------------------------------\n")

    print("Maximum Sub Array Iterative (Kadane's):")
    print(find_significant_energy_increase_iterative(ENERGY_LEVEL))
    print("\n--------------------------------------------------------\n")

    print("Square Matrix Multiplication using Strassens Algorithm:")
    print(square_matrix_multiply_strassens([[0, 1], [1, 1]],
                                           [[0, 1], [1, 1]]))
    print("\n--------------------------------------------------------\n")

    print("Naive Power Of Matrix with O(k) calls to Strassens Method:")
    print(power_of_matrix_navie([[0, 1], [1, 1]], 3))
    print("\n--------------------------------------------------------\n")

    print("Divide and Conquer Power Of Matrix "
          "with O(logk) calls to Strassens Method:")
    print(power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], 3))
    print("\n--------------------------------------------------------\n")


if __name__ == '__main__':
    test()

# ==============================================================
