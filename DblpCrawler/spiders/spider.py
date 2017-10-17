import scrapy
from ..items import DblpcrawlerItem
from ..rule_factory import RuleFactory
import re


class DblpCrawler(scrapy.Spider):

    name = 'dblpcrawler'
    max_iterations = 2
    iteration = 0

    def start_requests(self):
        urls = ["http://dblp.uni-trier.de/search/publ/inc?q=machine%20learning&h=30&f=0&s=yvpc"]

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        item = DblpcrawlerItem()

        # List of all entries on a page
        entry_list = response.xpath('//li[contains(@class, "entry")]')

        # Process each entry in the result list
        for entry in entry_list:

            # extract type of document we are looking at, and retrieve the corresponding rules
            rf = RuleFactory()
            entry_type = entry.xpath('./@class').extract_first()
            rule_list = rf.get_rule_list(entry_type)
            item['entry_type'] = entry_type.split(" ")[1]

            item['source_url'] = response.url

            item['title'] = entry.xpath(rule_list['title']).extract_first()

            # There may be more then 1 author, so this field is a list.
            # On the page each author has their own <span itemprop="author"> name </span> element, which we iterate over
            item['authors'] = []
            for author in entry.xpath(rule_list['author_iterator']):
                item['authors'].append(author.xpath(rule_list['author']).extract_first())

            item['doc_url'] = entry.xpath(rule_list['doc_url']).extract_first()

            item['isbn'] = entry.xpath(rule_list['isbn']).extract_first()

            item['publisher'] = entry.xpath(rule_list['publisher']).extract_first()

            item['publication_year'] = entry.xpath(rule_list['publication_year']).extract_first()

            item['publication'] = entry.xpath(rule_list['publication']).extract_first()


            yield item

        # If the current page has no result entries (empty page) then we have reached the end of the result
        # for a given start url. In this case we do not yield any more pages as this will only return empty pages.
        if len(entry_list) > 0 and self.iteration < self.max_iterations:
            # build the next page url by incrementing the f parameter in the url by 30.
            page_pattern = r'f=(\d*)'
            current_page_number = int(re.search(pattern=page_pattern, string=response.url).group(1))
            next_page_number = current_page_number + 30

            next_page = re.sub(page_pattern, "f={}".format(next_page_number), response.url)
            self.iteration = self.iteration + 1
            yield scrapy.Request(url=next_page, callback=self.parse)