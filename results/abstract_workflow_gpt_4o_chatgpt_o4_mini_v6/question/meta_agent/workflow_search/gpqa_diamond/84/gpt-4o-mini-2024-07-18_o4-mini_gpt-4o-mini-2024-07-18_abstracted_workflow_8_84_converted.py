async def forward_84(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    star_mass = 1.5  # in solar masses
    star_radius = 1.2  # in solar radii
    effective_temperature = 6300  # in Kelvin
    albedo = 0.3  # assumed constant for both planets
    distance = 1  # in AU, assumed for simplicity

    # Stage 1: Calculate equilibrium temperatures
    cot_instruction1 = "Sub-task 1: Calculate the equilibrium temperature for Planet1 using the formula Teq = Teff * (Rstar / (2 * d))^0.5."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, calculating Teq for Planet1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction2 = "Sub-task 2: Calculate the equilibrium temperature for Planet2 using the same formula as in subtask 1."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, calculating Teq for Planet2, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Determine the ratio of equilibrium temperatures
    cot_instruction3 = "Sub-task 3: Determine the ratio of the equilibrium temperatures between Planet1 and Planet2."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, answer1, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, calculating temperature ratio, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Evaluate choices against the calculated ratio
    cot_instruction4 = "Sub-task 4: Evaluate the choices provided (~1.05, ~0.98, ~0.53, ~1.30) against the calculated ratio to identify the closest match."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, evaluating choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer