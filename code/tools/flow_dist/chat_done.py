# -*- coding: utf-8 -*-
import utils

from conf import Configuration


if __name__ == '__main__':
    config = Configuration()

    if config.conf['chat_conf']['is_chat'] == 1:
        utils.chat(0, title=config.conf['chat_conf']['title'], content='（xxx）复制昨日流量分发结果!!!', url=config.conf['chat_conf']['url'])
