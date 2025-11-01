from Environment import BaseEnvironment

import pickle

with open('public_data_dict.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(data)


class PPO:
    def __init__(self):
        self.state_dimensions = 5 #t, SoC, price, pv & consumption
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        

environment = BaseEnvironment(data=data)

while True:
    chosen_action = float(input("what do u want to do: "))
    state, reward, done, reward2 = environment.step(chosen_action)
    t, SoC, price, pv, consumption = state
    print(f"at time {t} we have battery level {round(SoC * 100,2)}% for price {price}/KWh. The panels are producing {pv} kW and we're consuming {consumption} kW")
    print(f"observed {state}, {reward}, {done}, {reward2}")
    print(f"current battery level {environment.battery}")