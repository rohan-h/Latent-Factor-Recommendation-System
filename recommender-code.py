import random

import matplotlib.pyplot as plt
import numpy as np


def latent_factor_recommnder(data_path, regularization_factor=0.1, learning_rate=0.1, iterations=40, k=20):
    items = set()
    users = set()

    error_list = []

    with open(data_path) as f:
        for line in f:
            arr = line.split(",")
            items.add(int(arr[0]))
            users.add(int(arr[1]))

    items = sorted(items)
    users = sorted(users)

    m = len(items)
    n = len(users)

    Q = [[round(random.uniform(0, (5 / k) ** (1 / 2)), 2) for x in range(k)] for y in range(m)]
    P = [[round(random.uniform(0, (5 / k) ** (1 / 2)), 2) for x in range(k)] for y in range(n)]
    Q = np.array(Q)
    P = np.array(P)

    def error():
        first_part = 0
        p_u_l2normal_sum = 0
        q_i_l2normal_sum = 0

        # Computing first part of equation
        with open(data_path) as f:
            for line in f:
                i, u, riu = line.split(",")
                i = int(i)
                u = int(u)
                riu = int(riu)

                # multiplying Q[i] and P[u]^Transpose
                qpt = np.dot(Q[i], P[u].T)

                first_part += (riu - qpt) ** 2

        # Computing second part of equation
        for user in users:
            p_u_l2normal_sum += sum(map(lambda x: x ** 2, P[user]))

        for item in items:
            q_i_l2normal_sum += sum(map(lambda x: x ** 2, Q[item]))

        second_part = regularization_factor * (p_u_l2normal_sum + q_i_l2normal_sum)

        return first_part + second_part

    error_list.append(error())
    print("Initialization step: ", error())
    for iter in range(1, iterations + 1):
        # derivation of error
        with open(data_path) as f:
            for line in f:
                i, u, riu = line.split(",")
                i = int(i)
                u = int(u)
                riu = int(riu)

                qpt = round(np.dot(Q[i], P[u].T), 2)
                eui = 2 * (riu - qpt)
                old_Q = Q[i]
                Q[i] += learning_rate * ((eui * P[u]) - (2 * regularization_factor * Q[i]))
                P[u] += learning_rate * ((eui * old_Q) - (2 * regularization_factor * P[u]))

        error_list.append(error())

    for iter in range(1, iterations):
        print("Iteration ", iter, ": ", error_list[iter])

    print("Iteration: ", iterations, ", error:", error_list[iterations])
    return error_list


iterations = 40
k = 20
regularization_factor = 0.1  # λ
learning_rate = 0.018  # η
data_path = 'ratings.csv'

error_list = latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k)

iterAxis = list(range(0, iterations + 1))
plt.plot(iterAxis, error_list)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Iterations vs Error')
plt.xticks(size=iterations + 1)
plt.savefig("ErrorPlot.png")
plt.close()

