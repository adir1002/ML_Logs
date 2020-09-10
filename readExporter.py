
import csv
import time

import requests


def json_to_string(query_json):
    labels = list(query_json.keys())
    query_string = query_json['__name__']
    labels.remove('__name__')
    query_string += '{'
    query_string += ' '.join([label + '="' + query_json[label] + '",' for label in labels])
    query_string += '}'
    return query_string


def timestamp_query(query_values):
    timestamp_list = ['time']
    timestamp_list += [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[0])) for timestamp in query_values]
    return timestamp_list


def metric_query(query_values):
    return [timestamp[1] for timestamp in query_values]


def prometheus_to_csv():
    prometheus_url = "http://prometheus-hawkeye.westeurope.cloudapp.azure.com:9090"
    elastic_server_exporter = '{instance="elastic-hawkeye.westeurope.cloudapp.azure.com:9114"}'

    response = requests.get(prometheus_url + '/api/v1/query?query=' + elastic_server_exporter + '[6h]')
    queries_data = response.json()['data']['result']  # all the queries in the exporter

    with open("all_queries.csv", "w") as file:
        writer = csv.writer(file)
        write_time = True
        index = 0
        for query_data in queries_data:
            query_string = json_to_string(query_data['metric'])  # query string

            if write_time:
                time_list = timestamp_query(query_data['values'])
                writer.writerow(time_list)
                write_time = False

            query_list = metric_query(query_data['values'])
            writer.writerow([query_string] + query_list)


if __name__ == '__main__':
    prometheus_to_csv()
