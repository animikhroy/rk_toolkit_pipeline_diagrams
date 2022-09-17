'''
Filters
------------------
Built-in filters that comply with the
Filter interface.
'''
class RangeFilter():

    def __init__(self, min:float=0, max:float=1):
        self.min = min
        self.max = max

    def get_knobs(self):
        return {
            "min": self.min,
            "max": self.max
        }

    def set_knob(self, knb, v):
        if knb == "min":
            self.min = v
        elif knb == "max":
            self.max = v
        else:
            raise ValueError("No knob {}".format(knb))

    def filter(self, node):
        if not "value" in node or node["value"] is None:
            print("Warning: No value present for {}. Could not filter".format(node))
            return False

        return not (node["value"] > self.min and node["value"] <= self.max)

class FilterAll():

    def get_knobs(self):
        pass

    def set_knob(self, knb, v):
        pass

    def filter(self, node):
        return false


class FilterNone():

    def get_knobs(self):
        pass

    def set_knob(self, knb, v):
        pass

    def filter(self, node):
        return true
