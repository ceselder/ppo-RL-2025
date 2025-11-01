Here's the documentation for the Environment we provided you:


### **State**

*   t: ``int`` representing the hour of the day. Ranges between 0 and 23.
*   SoC: ``float`` representing the state-of-charge (percentage) of the battery. Goes from 0 (empty battery) to 1 (full battery)
*   price: ``float`` Electricity price addressed at time 't' in €/kWh.
*   pv: ``float`` Electricity production of the PhotoVoltatic panels at time 't' expressed in kW.
*   consumption: ``float`` Electricity consumption of the household at time 't' expressed in kW.

### **Action**

The action represents the amount of power which the battery is (dis-)charged with (kW). Each action is performed for an hour at each step.\
It is defined as an ``float`` that ranges between ``-power_capacity`` (max. discharge action) and ``power_capacity`` (max charge action). \
A negative value means the battery is being discharged, a positive value means the battery is being charged. A value of 0 means the battery is neither charging nor discharging.

### **Dynamics Function**

The battery (dis-)charges linearly at a rate decided by the action. 

### **Reward**
The reward is modeled as the cost of energy consumed in the given time step and calculated using the current price and real power consumed. \
The power consumed at each timestep consists of the sum of the battery action, the PV production, and the household consumption. \
If the Production is greater than the Consumption, the obtained profits gets divided by 4. \
The reward is defined as a ``float``.

The Environment code has 2 major functions that you will need:
- ``step(action: float) --> next_state, reward, done``. \
It performs the given action and returns 
the state of the environment right after the action is performed, 
the reward obtained from performing the action at the previous state, 
and a done (boolean) flag that becomes ``True`` when the considered day is over.
- ``reset(day_code: int) --> state``. \
Reset the environment to its initial conditions. 
Moreover, it loads the data of the day selected as input (``day_code``). 

The environment also contains other methods for normalizing the state and rewards.
- ``scale_state(state) --> norm_state``
- ``descale_state(norm_state) --> state``
- ``scale_reward(reward) --> norm_reward``
- ``descale_reward(norm_reward) --> reward``
- ``scale_action(action) --> norm_action``
- ``descale_action(norm_action) --> action``