import scrapy
from scrapy import Field


class DblpcrawlerItem(scrapy.Item):
    title = Field()
    authors = Field()
    doc_url = Field()
    isbn = Field()
    entry_type = Field()
    source_url = Field()
    publisher = Field()
    publication_year = Field()
    publication = Field()

    pass
