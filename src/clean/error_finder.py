import csv


# Write a list of string to file
def write_list(lst, file_):
    with open(file_,'w') as f:
        for line in lst:
            f.write(line)
            f.write('\n')


# Extract urls of error records and write to a file
def export_error_job_url(dataset_file):
    error_urls = []
    with open(dataset_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        total = 0
        for line in reader:
            total += 1
            if len(line) > 0 and line[0].strip()  and total > 3:
                error_urls.append(line[2])
    print('There are {} records'.format(total))
    print('There are {} error'.format(len(error_urls)))
    write_list(error_urls, 'error_url.txt')

# Extract duplicated URLs and write to a file
def export_duplicated_url(dataset_file):
    url_index = 2
    urls = []
    duplicated_urls = []
    with open(dataset_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) > 0:
                if line[url_index].strip() not in urls:
                    urls.append(line[url_index])
                else:
                    duplicated_urls.append(line[url_index])
    print('There are {} records'.format(len(urls)))
    print('There are {} duplicated URLs'.format(len(duplicated_urls)))
    write_list(duplicated_urls, 'duplicated_url.txt')


dataset = 'dataset.csv'
# Export error URLs
export_duplicated_url(dataset)
export_error_job_url(dataset)