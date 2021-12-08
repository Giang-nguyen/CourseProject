import csv
from os import remove


def replace_records(dataset1, dataset2):
    resolved_count = 0
    url_index = 2
    data1, data2 = [], []
    with open(dataset1, newline='', encoding='utf-8') as f1, open(dataset2, newline='', encoding='utf-8') as f2:
        data1 = list(csv.reader(f1))
        data2 = list(csv.reader(f2))
    for i in range(len(data1)):
        line1 = data1[i]
        if len(line1) > 0 and line1[0].strip()  and i > 3:
            for line2 in data2:
                if line2[url_index] == line1[url_index]:
                    data1[i] = line2
                    resolved_count += 1
    with open(dataset1, newline='', encoding='utf-8', mode='w') as f1:
        writer = csv.writer(f1)
        writer.writerows(data1)
    print('There are {} records resolved'.format(resolved_count))


def clean_error_records(dataset):
    removed_count = 0
    with open(dataset, newline='', encoding='utf-8') as f:
        data = list(csv.reader(f))
        i = 0
    while i < len(data):
        line = data[i]
        if len(line) > 0 and line[0].strip() and i > 3:
            removed_count += 1
            data.remove(line)
        i += 1
    with open(dataset, newline='', encoding='utf-8', mode='w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('There are {} records removed'.format(removed_count))


def remove_error_column(dataset):
    with open(dataset, newline='', encoding='utf-8') as f:
        data = list(csv.reader(f))
    for line in data:
        line.pop(0)
    with open(dataset, newline='', encoding='utf-8', mode='w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('Finish removing empty column.')


def change_columns(dataset):
    with open(dataset, newline='', encoding='utf-8') as f:
        data = list(csv.reader(f))
    for line in data:
        line.insert(0, line[-2])
        line.insert(0, line[-3])
        line.insert(0, line[-1])
        line.pop(-1)
        line.pop(-1)
        line.pop(-1)
    with open(dataset, newline='', encoding='utf-8', mode='w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print('Finish changing columns.')


def append(dataset1, dataset2):
    data1, data2 = [], []
    with open(dataset1, newline='', encoding='utf-8') as f1, open(dataset2, newline='', encoding='utf-8') as f2:
        data1 = list(csv.reader(f1))
        data2 = list(csv.reader(f2))
    data2.pop(0)
    data1.extend(data2)
    with open(dataset1, newline='', encoding='utf-8', mode='w') as f1:
        writer = csv.writer(f1)
        writer.writerows(data1)
    print('There are {} records appended.'.format(len(data2)))





dataset1 = 'dataset.csv'
dataset2 = 'dataset2.csv'

# Uncomment the corresponding method to clean data

#replace_records(dataset1, dataset2)
#clean_error_records(dataset1)
#remove_error_column(dataset2)
#change_columns('result.csv')
#append(dataset1, dataset2)