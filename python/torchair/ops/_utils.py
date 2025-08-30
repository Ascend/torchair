import torch


class TaggedEventFakeBidict:
    def __init__(self):
        self.global_tag_to_event = {}
    
    def __setitem__(self, tag, event):
        self.global_tag_to_event[tag] = event
    
    def keys(self):
        return self.global_tag_to_event.keys()
    
    def get(self, tag: str) -> torch.npu.Event:
        return self.global_tag_to_event.get(tag)
    
    def get_reverse(self, event: torch.npu.Event) -> str:
        for tag, tagged_event in self.global_tag_to_event.items():
            if tagged_event == event:
                return tag
        return None
    

class TaggedEventBidict(TaggedEventFakeBidict):
    def __init__(self):
        super().__init__()
        self.global_event_id_to_tag = {}
        
    def __setitem__(self, tag, event):
        self.global_tag_to_event[tag] = event
        self.global_event_id_to_tag[id(event)] = tag
        
    def get_reverse(self, event: torch.npu.Event) -> str:
        return self.global_event_id_to_tag.get(id(event))