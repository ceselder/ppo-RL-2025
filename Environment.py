import numpy as np

class Battery():
    def __init__(self, power_capacity: float=1, energy_capacity: float=1, SoC: float=0.5, efficiency: float=1):
        '''
        asd
        '''
        self.power_capacity = power_capacity
        self.energy_capacity = energy_capacity
        if (SoC > 1) or (SoC < 0):
            raise ValueError('State of Charge should be contained in [0, 1]')
        self.SoC = SoC
        self.efficiency = efficiency


    def charge(self, power: float, time_step_length: int):
        '''
        Charge the battery by applying the 'power input' for 'time_step_length' minutes.
        Positive power means charging. Negative power means discharging.
        '''

        real_power = min(max(power, -self.power_capacity), self.power_capacity)
        energy_delta = (real_power * time_step_length / 60)

        if power > 0:
            real_energy_delta = energy_delta * self.efficiency
        else:
            real_energy_delta = energy_delta * (2 - self.efficiency)

        new_SoC = min(max(self.SoC + real_energy_delta / self.energy_capacity, 0), 1)

        if new_SoC >= 1:
            energy_delta = (1 - self.SoC) * self.energy_capacity
            real_energy_delta = energy_delta
            self.SoC = 1
        elif new_SoC <= 0:
            energy_delta = (-self.SoC) * self.energy_capacity
            real_energy_delta = energy_delta
            self.SoC = 0
        else:
            self.SoC = new_SoC

        return energy_delta, real_energy_delta, self.SoC

class BaseEnvironment():
    def __init__(self, SoC: float=0.5, power_capacity: float=1, energy_capacity: float=1, efficiency: float=1, data: dict = None):
        self.num_minutes_per_step = 60    # Hyperparameter!
        self.initial_SoC = SoC
        self.data = data

        self.battery = Battery(SoC=SoC,
                               power_capacity=power_capacity,
                               energy_capacity=energy_capacity,
                               efficiency=efficiency)

        self.set_reward_bounds()
        self.reset()

    def set_reward_bounds(self):
        self.max_pv = 2.09
        self.max_cons = 5.73
        self.max_price = 0.12

        self.max_reward = (self.battery.power_capacity + self.max_pv) * self.max_price * 0.25
        self.min_reward = -(self.battery.power_capacity + self.max_cons) * self.max_price

    def step(self, action: float):
        energy_delta_battery, _, _ = self.battery.charge(action, self.num_minutes_per_step)
        energy_delta_household = self.household_consumption[min(self.t, len(self.household_consumption)-1)] * self.num_minutes_per_step / 60
        energy_delta_pv = self.pv_profile[min(self.t, len(self.pv_profile)-1)] * self.num_minutes_per_step / 60

        energy_balance = -energy_delta_household + energy_delta_pv - energy_delta_battery
        reward = energy_balance * self.prices[min(self.t, len(self.prices)-1)]

        if reward > 0:
          reward *= 0.25

        self.t += 1

        self.state = [self.t, float(self.battery.SoC),
                      self.prices[min(self.t, len(self.prices) - 1)],
                      self.pv_profile[min(self.t, len(self.pv_profile) - 1)],
                      self.household_consumption[min(self.t, len(self.household_consumption) - 1)]]
        # self.state = self.scale_state(self.state)

        if self.t >= 24:
            self.done = True

        return self.state, reward, self.done, reward

    def scale_state(self, state):
        norm_state = np.array([state[0] / 12 - 1,
                               2 * state[1] - 1,
                               2 * state[2] / self.max_price - 1,
                               2 * state[3] / self.max_pv - 1,
                               2 * state[4] / self.max_cons - 1])
        return norm_state

    def descale_state(self, norm_state):
        state = np.array([(norm_state[0] + 1) * 12,
                          (norm_state[1] + 1) / 2,
                          (norm_state[2] + 1) / 2 * self.max_price,
                          (norm_state[3] + 1) / 2 * self.max_pv,
                          (norm_state[4] + 1) / 2 * self.max_cons])

        return state

    def scale_reward(self, reward):
      return (reward - self.min_reward) / (self.max_reward - self.min_reward)

    def descale_reward(self, norm_reward):
      return norm_reward * (self.max_reward - self.min_reward) + self.min_reward

    def scale_action(self, action):
      return action / self.battery.power_capacity

    def descale_action(self, norm_action):
      return norm_action * self.battery.power_capacity

    def reset(self, day_code: int=100):
        self.t = 0

        self.prices = [0.06433, 0.04301, 0.03300, 0.01761, 0.00973, 0.01745, 0.05565, 0.08221,
                        0.12000, 0.10571, 0.09008, 0.07692, 0.07009, 0.06400, 0.06100, 0.04898,
                        0.05099, 0.07788, 0.09273, 0.10917, 0.09844, 0.08666, 0.06610, 0.03510]
        if day_code < 0:
            day_code = min(self.data.keys())
        self.pv_profile = list(self.data[day_code]['Production'])
        self.household_consumption = list(self.data[day_code]['Consumption'])


        self.battery = Battery(SoC=self.initial_SoC,
                               power_capacity=self.battery.power_capacity,
                               energy_capacity=self.battery.energy_capacity,
                               efficiency=self.battery.efficiency)

        self.state = [self.t, float(self.battery.SoC),
                      self.prices[min(self.t, len(self.prices)-1)],
                      self.pv_profile[min(self.t, len(self.pv_profile)-1)],
                      self.household_consumption[min(self.t, len(self.household_consumption)-1)]]
        # self.state = self.scale_state(self.state)
        self.done = False

        return self.state, {}