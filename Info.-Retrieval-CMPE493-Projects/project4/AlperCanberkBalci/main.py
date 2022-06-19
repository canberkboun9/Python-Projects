from operator import mul
from itertools import starmap


def mult_vector_matrix(vector, matrix):
    return [sum(starmap(mul, zip(vector, col))) for col in zip(*matrix)]


if __name__ == '__main__':

    word_dict = {}
    tp_rate = 0.15  # Teleportation Rate
    f = open("data.txt", "r")
    i = 0
    num_of_verts = 0
    for line in f:

        if i == 0:
            num_of_verts = int(line[10:])
            print("Number of Vertices", num_of_verts)

        if num_of_verts >= i >= 1:
            index_of_quote = line.index("\"")
            word = line[index_of_quote + 1:-2]

            word_dict[i - 1] = word

        i += 1

    adj_matrix_ = [0]*num_of_verts
    for i in range(num_of_verts):
        adj_matrix_[i] = [0]*num_of_verts


    f.close()
    f = open("data.txt", "r")

    j = 0
    read_edges = False
    # Step 1: Compute the Adjacency Matrix
    for line in f:

        if read_edges:
            j += 1
            line_tuple = line.split(" ")
            first_ = int(line_tuple[0]) - 1
            second_ = int(line_tuple[1]) - 1

            adj_matrix_[first_][second_] = 1
            adj_matrix_[second_][first_] = 1

        if line[0] + line[1] == "*E":
            read_edges = True

    num_of_edges = j
    # print(adj_matrix)
    # ndArray[row_index][column_index]
    # Step 2:
    # 2.1) If a row in Adjacency Matrix A has no 1, then replace each element by 1/Num of vertices (1/N)
    for i in range(num_of_verts):
        row_ = adj_matrix_[i]
        num_of_ones_in_row_ = row_.count(1)

        if num_of_ones_in_row_ == 0:
            for j in range(num_of_verts):
                adj_matrix_[i][j] = 1/num_of_verts  # step 2.1
        else:  # if there is 1 in a row
            for k in range(num_of_verts):
                adj_matrix_[i][k] = adj_matrix_[i][k] / num_of_ones_in_row_
                adj_matrix_[i][k] = adj_matrix_[i][k]*(1-tp_rate)
                # 2.2) Multiply the resulting matrix by 1-t
                adj_matrix_[i][k] = adj_matrix_[i][k] + tp_rate/num_of_verts
                # 2.3) Add tp_rate/num_of_verts to every element of the matrix to obtain P.

    print("sum: ", sum(adj_matrix_[0]))  # Sum of the row must be equal to 1. And it is.
    # We have obtained P.
    # Via Power Iteration Method, find PageRank Scores of each word.
    x = [1/num_of_verts]*num_of_verts  # fill this such that sum of elements will be = 1
    tolerable_error = 0.0000000001  # we want a reasonable error margin, but this parameter can be changed by hand!!!
    max_iteration = 20000  # matrix can be big for other inputs
    eig_old = 1.0  # for stochastic matrices...
    stop_condition = True  # true for while loop
    step = 1  # step counter
    while stop_condition:
        # x_new = np.dot(x, P)
        x = mult_vector_matrix(x, adj_matrix_)
        eig_new = max(x)  # if needed take absolute value.

        # Displaying Eigen value and Eigen Vector
        print('\nStep -', step)
        print('----------')
        print('Eigen Value = %0.15f' % eig_new)
        #for i in range(num_of_verts):
        #    print('%0.15f\t' % (x[i]))

        # check for max iteration
        step = step + 1
        if step > max_iteration:
            print('Does not converge')
            break

        # Calculating error
        error = abs(eig_new - eig_old)
        print('error=' + str(error))
        eig_old = eig_new
        stop_condition = error > tolerable_error

    x_dict = {}
    for i in range(num_of_verts):

        x_dict[i] = x[i]  # list comprehension with sorting by value.
        x_dict = {k: v for k, v in sorted(x_dict.items(), key=lambda item: item[1], reverse=True)}

    resulting_list = list(x_dict.values())
    # printing the result
    resulting_dict = {}
    for i in range(num_of_verts):
        word_ = word_dict[i]
        score = x_dict[i]
        resulting_dict[word_] = score

    resulting_dict = {k: v for k, v in sorted(resulting_dict.items(), key=lambda item: item[1], reverse=True)}
    print("\nResulting Scores of PageRank:\n")
    count = 0
    for k, v in resulting_dict.items():
        print(k, ":", v)
        count += 1
        if count == 50:
            print("Above the top 50 people as well as their PageRank scores.")