import csv


# Write a list of string to file
def write_list(lst, file_):
    with open(file_,'w') as f:
        for line in lst:
            f.write(line)
            f.write('\n')


# Extract urls of error records and write to a file
def export_error_job_url():
    dataset_file = 'dataset.csv'
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


# Export urls of error records to re-crawl
export_error_job_url()