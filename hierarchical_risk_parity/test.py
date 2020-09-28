c_items = [[8, 3, 9, 2, 10, 1, 7, 5, 4, 6]]

c_items_arr = []

for i in c_items:
    print('i:', i)
    equation = ((0, int(len(i)/2)), (int(len(i)/2), len(i)))
    print('equation:', equation)
    for j, k in equation:
        print('j:', j)
        print('k:', k)
        if len(i) > 1:
            c_items_arr.append(i[j:k])


# c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)/2), (len(i)/2, len(i))) if len(i) > 1]

print(c_items_arr)
