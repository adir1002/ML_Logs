import csv
import requests
import sys
import datetime
import time

prometheus_url = "http://prometheus-hawkeye.westeurope.cloudapp.azure.com:9090"
elastic_server_exporter = '{instance="elastic-hawkeye.westeurope.cloudapp.azure.com:9114"}'


def JsonToString(query_json):
    labels = list(query_json.keys())
    query_string = query_json['__name__']
    labels.remove('__name__')
    query_string += '{'
    for label in labels:
        query_string += label + '="' + query_json[label] + '",'
    query_string += '}'
    return query_string


def TimeStampQuery(query_answer):
    timestamp_list = ['time']
    for timestamp in query_answer.json()['data']['result'][0]['values']:
        timestamp_list.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[0])))
    return timestamp_list


def MetricQuery(query_answer):
    metric_list = []
    for timestamp in query_answer.json()['data']['result'][0]['values']:
        metric_list.append(timestamp[1])
    return metric_list


response = requests.get(prometheus_url + '/api/v1/query?query=' + elastic_server_exporter)
queries_data = response.json()['data']['result']  # all the queries in the exporter

with open("a.csv", "w") as file:
    writer = csv.writer(file)
    writeTime = True
    for query_data in queries_data:
        query_string = JsonToString(query_data['metric'])  # query string
        answer = requests.get(prometheus_url + '/api/v1/query?query=' + query_string + '[6h]')

        if writeTime:
            time_metric = answer.json()['data']['result'][0]['values'][0][0]
            time_list = TimeStampQuery(answer)
            writer.writerow(time_list)
            writeTime = False

        query_list = MetricQuery(answer)
        writer.writerow([query_string] + query_list)
