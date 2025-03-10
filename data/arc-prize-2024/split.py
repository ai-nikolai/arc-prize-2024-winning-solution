# ('00576224', {'test': [{'input': [[3, 2], [7, 8]]}], 'train': [{'input': [[8, 6], [6, 4]], 'output': [[8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4], [6, 8, 6, 8, 6, 8], [4, 6, 4, 6, 4, 6], [8, 6, 8, 6, 8, 6], [6, 4, 6, 4, 6, 4]]}, {'input': [[7, 9], [4, 3]], 'output': [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3], [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4], [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]]}]})


import random
import json

if __name__=="__main__":
    with open("arc-agi_evaluation_challenges.json") as f:
        data1 = json.load(f)

    with open("arc-agi_evaluation_solutions.json") as f:
        data2 = json.load(f)

    new_data1 = {}
    new_data2 = {}

    new_data3 = {}
    new_data4 = {}
    counter = 0
    for key, value in data1.items():

        if counter<300 and random.random()>0.25:
            counter += 1
            new_data1[key] = value
            new_data2[key] = data2[key]
        else:
            new_data3[key] = value
            new_data4[key] = data2[key]            

    print(f"We have {len(new_data1)}, {len(new_data2)} eval data points & {len(new_data3)},{len(new_data4)} test data points.")
    
    with open("arc-agi-fixed-evaluation_challenges-v1.json", "w") as f:
        json.dump(new_data1,f)
    with open("arc-agi-fixed-evaluation_solutions-v1.json", "w") as f:
        json.dump(new_data2,f)
    with open("arc-agi-fixed-test_challenges-v1.json", "w") as f:
        json.dump(new_data3,f)
    with open("arc-agi-fixed-test_solutions-v1.json", "w") as f:
        json.dump(new_data4,f)
