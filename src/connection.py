from py2neo import Graph
import configparser
import os

def get_conf():
    conf = configparser.ConfigParser()
    main_path = os.getcwd()
    conf.read(main_path + '/configurations/configurations.ini')
    return conf

def connect(conf):
    graph = Graph(conf.get("DATABASE","host"), auth=(conf.get("DATABASE","username"), conf.get("DATABASE","password")))
    return graph
