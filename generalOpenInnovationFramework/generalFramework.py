from LandScape import *


class Firm:

    def __init__(self, N, decision_num, landscape=None):

        self.N = N
        self.state = np.random.choice([0, 1], N).tolist()
        self.decision_space = np.random.choice(N, decision_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)
        self.landscape = landscape

    def independent_search(self, ):
        base_state = list(self.state)

        c = np.random.choice(self.decision_space)

        base_state[c] ^= 1

        if self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.knowledge_space
        ):
            self.state = list(base_state)

    def pickup(self, alternatives):

        base_state = list(self.state)

        for alter in alternatives:
            if self.landscape.query_cog_fitness(
                alter, self.knowledge_space
            ) > self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
            ):
                base_state = list(alter)

        self.state = list(base_state)


class Agent:

    def __init__(self, N, knowledge_num, landscape=None):

        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.decision_space = np.random.choice(self.N, knowledge_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)

        self.landscape = landscape

    def independent_search(self, ):

        base_state = list(self.state)

        c = np.random.choice(self.decision_space)

        base_state[c] ^= 1

        if self.landscape.query_cog_fitness(
            base_state, self.knowledge_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.knowledge_space
        ):
            self.state = list(base_state)


class Industry:

    def __init__(self, N, firm_knowledge, agent_knowledge, agent_num, landscape,
                 crowd2firm_probability, firm2crowd_probability
                 ):

        self.firm = Firm(N, firm_knowledge, landscape)
        self.agents = [Agent(N, agent_knowledge, landscape) for cur in range(agent_num)]
        for agent in self.agents:
            agent.state = list(self.firm.state)

        self.crowd2firm_probability = crowd2firm_probability
        self.firm2crowd_probability = firm2crowd_probability

        # tag takes 0 if firm does search else 1
        self.tag = 0
        self.firm_timestamp = [0]
        self.crowd_timestamp = []

    def adaptation(self, step, last_period=False):

        if self.tag==0:
            self.firm.independent_search()

            if np.random.uniform(0, 1)<self.firm2crowd_probability:
                self.tag = 1
                for agent in self.agents:
                    agent.state = list(self.firm.state)
                if not last_period:
                    self.crowd_timestamp.append(step+1)
        else:
            for agent in self.agents:
                agent.independent_search()

            if np.random.uniform(0, 1)<self.crowd2firm_probability or last_period:
                self.tag = 0
                self.firm.pickup([agent.state for agent in self.agents])
                self.firm_timestamp.append(step+1)


def simulation(N, k, land_num, agent_num, period, firm_knowledge, agent_knowledge,
               crowd2firm_probability, firm2crowd_probability
               ):

    ress_fitness = []
    ress_firm_interval = []
    ress_crowd_interval = []

    for repeat in range(land_num):

        res_fitness = []
        res_firm_interval = []
        res_crowd_interval = []

        np.random.seed(None)

        landscape = LandScape(N, k, None, None)
        landscape.initialize()

        industry = Industry(
            N, firm_knowledge, agent_knowledge, agent_num, landscape, crowd2firm_probability,
            firm2crowd_probability
        )

        for step in range(period):
            industry.adaptation(step, step==period-1)
            res_fitness.append(industry.firm.landscape.query_fitness(industry.firm.state))

        cur = 0

        while cur<len(industry.firm_timestamp):

            if cur >= len(industry.crowd_timestamp):
                res_firm_interval.append(30-industry.firm_timestamp[cur])
                break
            else:
                res_firm_interval.append(industry.crowd_timestamp[cur]-industry.firm_timestamp[cur])
                res_crowd_interval.append(industry.firm_timestamp[cur+1]-industry.crowd_timestamp[cur])
                cur += 1

        ress_fitness.append(res_fitness)
        ress_crowd_interval.append(np.mean(res_crowd_interval) if len(res_crowd_interval) > 0 else -1)
        ress_firm_interval.append(np.mean(res_firm_interval) if len(res_firm_interval) > 0 else -1)

    return ress_fitness, ress_crowd_interval, ress_firm_interval











