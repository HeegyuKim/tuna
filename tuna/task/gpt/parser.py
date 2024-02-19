
class BaseParser:

    def parse_item(self, item):
        pass


class HHRLHFParser(BaseParser):

    def parse_item(self, item):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        return convs

class PKUSafeRLHFParser(BaseParser):

    def parse_item(self, item):
        convs = item["context"] + [item["instruction"], item["chosen"]]
        return convs