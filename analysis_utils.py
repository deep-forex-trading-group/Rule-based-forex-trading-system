from pandas import Series
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def difference_rate(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = (dataset[i] - dataset[i - interval])/dataset[i - interval]
        diff.append(value)
    return Series(diff)
    