import sqlite3
import re

_crawler_data = '../../data/crawler_sqlite.db'

def get_info(title):
    # Persistent connection would probably be more efficient, but this seems to work well enough.
    with sqlite3.connect(_crawler_data) as conn:

        # Turn result into dict with {db column name: entry}
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get any information we have in the crawled data.
        # Uses the LIKE operator since there may be small differences. For example, the title in
        # DBLP often has punctuation at the end, while the stored titles do not.
        q_title = title + "%"
        res = c.execute("""SELECT * FROM DATA WHERE title like ? """, (q_title,)).fetchone()

        # Exclude these columns since they are not relevant/interesting for the query
        exclude = ["title", "entry_type"]
        if res is not None:
            res_string = ""
            for key in res.keys():
                if res[key] != "none" and key not in exclude:
                    res_string = res_string + " {}: {}".format(key, res[key])
            # Remove the ugly brackets around the authors
            return re.sub("\[|\]", "", res_string.strip())
        else:
            return "Paper not in crawled Data"