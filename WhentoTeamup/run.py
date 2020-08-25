from simulation import *
import multiprocessing
import pickle

# baseline random team up at beginning


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()

    jobs = []
    for k in range(4):
        p = multiprocessing.Process(
            target=simulation, args=(
                16, k, None, None, 200, 1, 200, return_dic, True, False, 20, False, 0
            )
        )
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    res = return_dic.values()
    keys = return_dic.keys()
    res_dic = {keys[i]:res[i] for i in range(len(res))}
    f = open("./baseline.txt","wb")
    pickle.dump(res_dic,f)
    f.close()