# playing with the data to see how many users are there.

f = open("user_ids.txt", "r")

text = f.read()
ids = text.split("\n")

user_dict = {}

for user_id in ids:
	if user_id in user_dict:
		user_dict[user_id] += 1
	else: 
		user_dict[user_id] = 1

print len(user_dict)

import operator
sorted_dict = sorted(user_dict.items(), key=operator.itemgetter(1))[::-1]
print sorted_dict

print user_dict["9512"]
print user_dict["7970"]
print user_dict["9974"]

