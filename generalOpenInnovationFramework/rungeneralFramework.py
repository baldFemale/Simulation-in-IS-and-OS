from generalFramework import *
import multiprocessing
import pickle

N = 10
land_num = 500
period = 50

agentNum = 10

# to_file
file_path = "./output_generalFramework_{agentNum}".format(
    agentNum=agentNum,
)


def parallel_simulation(idx, return_dic, k, crowd2firm_probability, firm2crowd_probability):
    ress = []

    for firm_knowledge in [4, 6, 8]:
        for agent_knowledge in [4, 6, 8]:
            print(firm_knowledge, agent_knowledge)
            ress.append(simulation(N, k, land_num, agentNum, period, firm_knowledge, agent_knowledge,
                                   crowd2firm_probability, firm2crowd_probability))

    return_dic[idx] = ress


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dic = manager.dict()
    print(
        """
        def simulation(return_dic, idx, N, k, land_num, agent_num, period, firm_knowledge, agent_knowledge,
               crowd2firm_probability, firm2crowd_probability
               )
        """
    )

    jobs = []

    index = 0

    for crowd2firm_probability in [0, 0.25, 0.5, 0.75, 1]:
        for firm2crowd_probability in [0, 0.25, 0.5, 0.75, 1]:
            for k in range(0, 10, 2):

                p = multiprocessing.Process(
                    target=parallel_simulation, args=(
                        return_dic, index, k, crowd2firm_probability, firm2crowd_probability
                    )
                )

                jobs.append(p)
                p.start()

                index += 1

    for proc in jobs:
        proc.join()

    res_dic = {}

    for k in return_dic.keys():
        res_dic[k] = return_dic[k]

    f = open(file_path, "wb")
    pickle.dump(res_dic, f)
    f.close()
