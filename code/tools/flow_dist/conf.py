# -*- coding: utf-8 -*-
import argparse
import utils


class Configuration(object):
    def __init__(self):
        args = Configuration.args_parse()

        self.date = args.yesterday
        self.year = args.yesterday_year
        self.month = args.yesterday_month
        self.day = args.yesterday_day

        config = utils.get_config(args.config_path)
        conf = config.Configuration

        for k, _ in conf['check_conf']['hdfs'].items():
            conf['check_conf']['hdfs'][k] = conf['check_conf']['hdfs'][k].replace('${yesterday_year}', self.year)
            conf['check_conf']['hdfs'][k] = conf['check_conf']['hdfs'][k].replace('${yesterday_month}', self.month)
            conf['check_conf']['hdfs'][k] = conf['check_conf']['hdfs'][k].replace('${yesterday_day}', self.day)

        self.conf = conf


    @staticmethod
    def args_parse():
        parser = argparse.ArgumentParser(description='flow distribute')
        parser.add_argument('--yesterday', default='', type=str, help='昨天（日期）')
        parser.add_argument('--yesterday_year', default='', type=str, help='昨天（年）')
        parser.add_argument('--yesterday_month', default='', type=str, help='昨天（月）')
        parser.add_argument('--yesterday_day', default='', type=str, help='昨天（日）')
        parser.add_argument('--config_path',
                            default='config/distribute.yaml', type=str,
                            help='配置文件路径')
        args = parser.parse_args()
        return args
