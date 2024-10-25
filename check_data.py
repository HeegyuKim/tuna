import datasets as hfds


ds = hfds.load_dataset('iknow-lab/hermes-function-calling-v1-ko', split='train')


clean_map = {
    "도구": "tool",
    "토큰": "user",
    "사용자": "user",
    "Assistant": "assistant",
    "도움말": "assistant",
    "사람": "user",
    "조수": "assistant",
    "도우미": "assistant",
    "어시스턴트": "assistant",
    "유저": "user",
    "시스템": "system",
}
def map_item(item):
    for uttr in item["ko_conversations"]:
        if uttr["role"] in clean_map:
            uttr["role"] = clean_map[uttr["role"]]
    return item

ds = ds.map(map_item)

roles = set()
for item in ds:
    for uttr in item["ko_conversations"]:
        if uttr["role"] not in roles:
            print(uttr["role"], "/", uttr["content"])
            print("****" * 10)
            roles.add(uttr["role"])
print(roles)

ds = ds.remove_columns(["conversations"])
ds = ds.rename_column("ko_conversations", "messages")

ds.push_to_hub("iknow-lab/hermes-function-calling-v1-ko-cleaned")