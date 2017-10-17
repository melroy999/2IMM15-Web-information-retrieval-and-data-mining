# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import sqlite3
import os


class DblpcrawlerPipeline(object):

    def open_spider(self, spider):
        self.f = open("log.txt", "w")

    def close_spider(self, spider):
        self.f.close()

    def process_item(self, item, spider):
        for key in item.keys():
            if item[key] is None:
                item[key] = "none"

        for key in item.keys():
            if key in item:
                self.f.write("{}: {} \n".format(key, item[key]))
        self.f.write("\n")
        return item


class DblpcrawlerSqlitePipeline(object):

    database = os.path.abspath(os.path.dirname(__file__) + '/data/crawler_sqlite.db')

    def open_spider(self, spider):
        self.conn = sqlite3.connect(database=self.database)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS DATA 
                    (title TEXT PRIMARY KEY ON CONFLICT IGNORE NOT NULL, author TEXT, doc_url TEXT, isbn TEXT, 
                    publisher TEXT, publication_year TEXT, publication TEXT,
                    entry_type TEXT, source_url TEXT 
                    );""")

    def close_spider(self, spider):
        self.conn.close()

    def process_item(self, item, spider):
        for key in item.keys():
            if item[key] is None:
                item[key] = "none"

        t = (item['title'], str(item['authors']), item['doc_url'], item['isbn'], item['publisher'],
             item['publication_year'], item['publication'], item['entry_type'], item['source_url'])

        self.conn.execute("""INSERT INTO DATA VALUES (?,?,?,?,?,?,?,?,?)""", t)
        self.conn.commit()

if __name__ == "__main__":
    item = {
        'title': 'A Machine-Learning-Driven Sky Model.',
        'source_url': 'http://dblp.uni-trier.de/search/publ/inc?q=machine%20learning&h=30&f=60&s=yvpc',
        'publication_year': '2017',
        'isbn': 'none',
        'publisher': 'none',
        'publication': 'IEEE Computer Graphics and Applications',
        'entry_type': 'article',
        'authors': "['Pynar Satylmys', 'Thomas Bashford-Rogers', 'Alan Chalmers', 'Kurt Debattista']",
        'doc_url': 'https://doi.org/10.1109/MCG.2016.67'
    }
    dbc = DblpcrawlerSqlitePipeline()
    dbc.open_spider("")

    t = (item['title'], str(item['authors']), item['doc_url'], item['isbn'], item['publisher'],
         item['publication_year'], item['publication'], item['entry_type'], item['source_url'])

    q = """INSERT INTO DATA VALUES (?,?,?,?,?,?,?,?,?)"""
    dbc.conn.execute(q,t)
    dbc.conn.commit()
    dbc.close_spider("")