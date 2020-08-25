import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from scipy.stats import ttest_ind


def createInfluenceMatrix(N, K, K_within=None, K_between=None):
    IM = np.eye(N)
    if not K_within:
        for i in range(N):
            probs = [1 / (N - 1)] * i + [0] + [1 / (N - 1)] * (N - 1 - i)
            ids = np.random.choice(N, K, p=probs, replace=False)
            for index in ids:
                IM[i][index] = 1
    else:
        for i in range(N):
            if i // (N // 2) < 1:
                within = [j for j in range(N // 2)]
                between = [j for j in range(N // 2, N)]
                probs = [1 / (N // 2 - 1)] * i + [0] + [1 / (N // 2 - 1)] * (N // 2 - 1 - i)
                ids_within = np.random.choice(within, K_within, p=probs, replace=False)
                ids_between = np.random.choice(between, K_between, replace=False)
                for index in ids_within:
                    IM[i][index] = 1
                for index in ids_between:
                    IM[i][index] = 1

            else:
                within = [j for j in range(N // 2, N)]
                between = [j for j in range(N // 2)]
                probs = [1 / (N // 2 - 1)] * (i - N // 2) + [0] + [1 / (N // 2 - 1)] * (N - 1 - i)
                ids_between = np.random.choice(between, K_between, replace=False)
                ids_within = np.random.choice(within, K_within, p=probs, replace=False)
                for index in ids_within:
                    IM[i][index] = 1
                for index in ids_between:
                    IM[i][index] = 1

    IM_dic = defaultdict(list)
    for i in range(len(IM)):
        for j in range(len(IM[0])):
            if i == j or IM[i][j] == 0:
                continue
            else:
                IM_dic[i].append(j)
    return IM, IM_dic


def createFitnessConfig(IM):
    FC = defaultdict(dict)
    for row in range(len(IM)):
        k = int(sum(IM[row]))
        for i in range(pow(2,k)):
            FC[row][i] = np.random.uniform(0,1)
    return FC


def calculate_Fitness(state,IM_dic,FitnessConfig):
    res = 0.0
    for i in range(len(state)):
        dependency = IM_dic[i]
        bin_index = "".join([str(state[j]) for j in dependency])
        if state[i]==0:
            bin_index = "0" + bin_index
        else:
            bin_index = "1" + bin_index
        index = int(bin_index,2)
        res+=FitnessConfig[i][index]
    return res/len(state)


def calculateCogFitness(state,IM_dic,FitnessConfig,ids):
    cf = []
    l = len(ids)
    if l==0:
        return calculate_Fitness(state,IM_dic,FitnessConfig)
    for i in range(pow(2,l)):
        bit = bin(i)[2:]
        if len(bit)<l:
            bit = "0"*(l-len(bit))+bit
        temp_state = list(state)
        for j,k in enumerate(ids):
            temp_state[k] = int(bit[j])
        cf.append(calculate_Fitness(temp_state,IM_dic,FitnessConfig))
    return sum(cf)*1.0/(pow(2,l))


def initializeState(N):
    return np.random.choice([0,1],N)


def searchCache(state, IM_dic, FC, remainders, cache):
    stateStr = "".join(str(state[i]) if i not in remainders else "*" for i in range(len(state)))
    if stateStr in cache:
        return "",cache[stateStr]
    else:
        v = calculateCogFitness(state,IM_dic,FC,remainders)
        return stateStr, v


def indepenSearch(state, agentType, N, fitness_value, decision_space, IM_dic, FC, cache):
    if agentType=="A":
        choice = np.random.choice([i for i in range(N//2)],1)[0]
    else:
        choice = np.random.choice([i for i in range(N//2, N)],1)[0]
    temp = list(state)
    temp[choice] = temp[choice]^1
    inCache, temp_value = searchCache(temp, IM_dic, FC, [i for i in range(N) if i not in decision_space], cache)
    if temp_value>fitness_value:
        return temp_value, choice, inCache
    else:
        return fitness_value, None, inCache


def simulation(N, K, K_within, K_between, agentNum, landNum, period, return_dic, randomTeamup=False,
               synchronization=True, learningInterval=20, learning=True, teamupTime=100):
    IM, IM_dic = createInfluenceMatrix(N, K, K_within, K_between)
    ress = []
    teamsList = []

    for landscape in range(landNum):

        cache = {}
        res = []

        # initialization
        FC = createFitnessConfig(IM)
        agents = ["A"] * agentNum + ["B"] * agentNum
        states = np.zeros((agentNum * 2, N)).astype(np.int16)
        for i in range(agentNum * 2):
            states[i] = initializeState(N)
        fitnessValues = []
        decisionSpaces = []

        # normalization
        non_Fitness = []
        for i in range(pow(2, N)):
            temp = bin(i)[2:]
            if len(temp) < N:
                temp = "0" * (N - len(temp)) + temp
            temp = [int(i) for i in temp]
            non_Fitness.append(calculate_Fitness(temp, IM_dic, FC))
        normalizor = np.max(non_Fitness)

        for i in range(agentNum * 2):
            if i < agentNum:
                decisionSpaces.append([j for j in range(N // 2)])
                inCache, v = searchCache(states[i], IM_dic, FC, [cur + N // 2 for cur in range(N // 2)], cache)
                if len(inCache) > 0:
                    cache[inCache] = v
                fitnessValues.append(v)
            else:
                decisionSpaces.append([j + N // 2 for j in range(N // 2)])
                inCache, v = searchCache(states[i], IM_dic, FC, [cur for cur in range(N // 2)], cache)
                if len(inCache) > 0:
                    cache[inCache] = v
                fitnessValues.append(v)

        teams = {i: i for i in range(agentNum * 2)}

        for step in range(period):

            for i in range(agentNum * 2):

                # exploration

                if not teams[i]:
                    continue

                proposal = None

                if teams[i] == i:
                    v, c, inCache = indepenSearch(
                        states[i], agents[i], N, fitnessValues[i], decisionSpaces[i], IM_dic, FC, cache
                    )
                    if len(inCache) > 0:
                        cache[inCache] = v
                    if c:
                        states[i][c] = states[i][c] ^ 1
                        fitnessValues[i] = v
                else:
                    if agents[i] != agents[teams[i]]:
                        # proposal <value, new choice, inCache>
                        proposal = [
                            indepenSearch(states[i], agents[i], N, fitnessValues[i], decisionSpaces[i],
                                          IM_dic, FC, cache),
                            indepenSearch(states[i], agents[teams[i]], N, fitnessValues[teams[i]],
                                          decisionSpaces[teams[i]], IM_dic, FC, cache)
                        ]

                        for p in proposal:
                            if len(p[-1]) > 0:
                                cache[p[-1]] = p[0]

                # synchronization
                if synchronization:
                    if proposal:
                        temp_state = list(states[i])
                        if proposal[0][1]:
                            temp_state[proposal[0][1]] = temp_state[proposal[0][1]] ^ 1
                        if proposal[1][1]:
                            temp_state[proposal[1][1]] = temp_state[proposal[1][1]] ^ 1
                        inCacheA, vA = searchCache(temp_state, IM_dic, FC, [
                            cur for cur in range(N) if cur not in decisionSpaces[i]
                        ], cache)
                        if len(inCacheA) > 0:
                            cache[inCacheA] = vA
                        inCacheB, vB = searchCache(temp_state, IM_dic, FC, [
                            cur for cur in range(N) if cur not in decisionSpaces[teams[i]]
                        ], cache)
                        if len(inCacheB) > 0:
                            cache[inCacheB] = vB
                        if vA >= fitnessValues[i] and vB >= fitnessValues[teams[i]]:
                            states[i] = temp_state
                            fitnessValues[i] = vA
                            fitnessValues[teams[i]] = vB
                else:
                    if proposal:
                        if proposal[0][1]:
                            states[i][proposal[0][1]] = states[i][proposal[0][1]] ^ 1
                            fitnessValues[i] = proposal[0][0]
                        if proposal[1][1]:
                            states[i][proposal[1][1]] = states[i][proposal[1][1]] ^ 1
                            fitnessValues[teams[i]] = proposal[1][0]

            if step == teamupTime:
                if randomTeamup:
                    rank = np.random.choice([cur for cur in range(agentNum, agentNum * 2)], agentNum, replace=False)
                    for i in range(agentNum):
                        teams[i] = rank[i]
                        teams[rank[i]] = None
                        states[i][N // 2:] = states[rank[i]][N // 2:]
                else:
                    tempA = sorted(
                        [(fitnessValues[i], i) for i in range(agentNum * 2) if agents[i] == "A"],
                        key=lambda x: -x[0]
                    )
                    tempB = sorted(
                        [(fitnessValues[i], i) for i in range(agentNum * 2) if agents[i] == "B"],
                        key=lambda x: -x[0]
                    )
                    while len(tempA) > 0 and len(tempB) > 0:
                        # for A
                        fitnessValue, index = tempA.pop(0)
                        temp_state = list(states[index])
                        temp_state[N // 2:N] = states[tempB[0][1]][N // 2:N]
                        maxValue = calculate_Fitness(temp_state, IM_dic, FC)
                        maxchoice = 0
                        for i in range(1, len(tempB)):
                            temp_state[N // 2:N] = states[tempB[i][1]][N // 2:N]
                            currentValue = calculate_Fitness(temp_state, IM_dic, FC)
                            if currentValue > maxValue:
                                maxValue = currentValue
                                maxchoice = i
                        teamMateValue, teamMate = tempB.pop(maxchoice)
                        teams[index] = teamMate
                        teams[teamMate] = None
                        states[index][N // 2:N] = states[teamMate][N // 2:N]
                        fitnessValues[index] = calculate_Fitness(states[index], IM_dic, FC)
                        # for B
                        fitnessValue, index = tempB.pop(0)
                        temp_state = list(states[index])
                        temp_state[:N // 2] = states[tempA[0][1]][0:N // 2]
                        maxValue = calculate_Fitness(temp_state, IM_dic, FC)
                        maxchoice = 0
                        for i in range(1, len(tempA)):
                            temp_state[:N // 2] = states[tempA[i][1]][:N // 2]
                            currentValue = calculate_Fitness(temp_state, IM_dic, FC)
                            if currentValue > maxValue:
                                maxValue = currentValue
                                maxchoice = i
                        teamMateValue, teamMate = tempA.pop(maxchoice)
                        teams[teamMate] = index
                        teams[index] = None
                        states[teamMate][N // 2:N] = states[index][N // 2:N]
                        fitnessValues[teamMate] = calculate_Fitness(states[teamMate], IM_dic, FC)

            if learning and step > teamupTime and (step - teamupTime) % learningInterval == 0 and (
                    step - teamupTime) // learningInterval <= N // 2:
                for i in range(agentNum * 2):
                    if teams[i] and teams[i] != i:
                        newKnowledgeA = np.random.choice(
                            decisionSpaces[teams[i]], 1, replace=False
                        )
                        newKnowledgeB = np.random.choice(
                            decisionSpaces[i], 1, replace=False
                        )
                        decisionSpaces[i] = decisionSpaces[i] + newKnowledgeA.tolist()
                        decisionSpaces[teams[i]] = decisionSpaces[teams[i]] + newKnowledgeB.tolist()
                        fitnessValues[i] = calculateCogFitness(
                            states[i], IM_dic, FC, [cur for cur in range(N) if cur not in decisionSpaces[i]]
                        )
                        fitnessValues[teams[i]] = calculateCogFitness(
                            states[i], IM_dic, FC, [cur for cur in range(N) if cur not in decisionSpaces[teams[i]]]
                        )

            tempFitness = [calculate_Fitness(states[i], IM_dic, FC) / normalizor for i in range(len(states))]
            #             print(normalizor)
            #             print(np.mean(tempFitness[:agentNum]))
            res.append(tempFitness)
        ress.append(res)
        teamsList.append(teams)
    return_dic[K] = [np.array(ress), teamsList, teamupTime]

