TestList = [
    [1,2,3],
    [2]
]

var = any(
    len(l)>4 for l in TestList
)
print(var)