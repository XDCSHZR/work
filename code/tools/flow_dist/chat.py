# -*- coding: utf-8 -*-
import utils

from conf import Configuration


if __name__ == '__main__':
    config = Configuration()

    if config.conf['chat_conf']['is_chat'] == 1:
        utils.chat(1, title=config.conf['chat_conf']['title'], content='Please check log for errors!!!', url=config.conf['chat_conf']['url'])
