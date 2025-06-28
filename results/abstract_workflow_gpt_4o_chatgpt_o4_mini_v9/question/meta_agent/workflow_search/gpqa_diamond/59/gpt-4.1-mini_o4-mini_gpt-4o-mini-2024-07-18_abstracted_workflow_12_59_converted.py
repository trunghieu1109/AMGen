async def forward_59(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Extract parameters and compute fundamental values
    
    # Sub-task 1: Identify and extract all given quantitative parameters and conditions
    cot_instruction_1 = (
        "Sub-task 1: Identify and extract all given quantitative parameters and conditions from the query, "
        "including spacecraft speed (0.99999987*c), astronaut age (22 years), alien average lifetime (150 solar years), "
        "and the destination (Earth). This sets the foundation for all subsequent calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine the distance between the Large Magellanic Cloud and Earth in light years
    cot_instruction_2 = (
        "Sub-task 2: Determine the distance between the Large Magellanic Cloud and Earth in light years, "
        "as this is necessary to calculate travel time at relativistic speeds. This uses the context that the spacecraft is traveling from the Large Magellanic Cloud to Earth."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining distance, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Calculate the Lorentz factor (gamma) for the spacecraft traveling at 0.99999987*c
    cot_instruction_3 = (
        "Sub-task 3: Calculate the Lorentz factor (gamma) for the spacecraft traveling at 0.99999987*c, "
        "which is essential for applying time dilation effects to the astronauts experienced travel time."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating Lorentz factor, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Sub-task 4: Compute the travel time from the Large Magellanic Cloud to Earth as measured in the Earth frame
    cot_instruction_4 = (
        "Sub-task 4: Compute the travel time from the Large Magellanic Cloud to Earth as measured in the Earth frame (stationary frame), "
        "using the distance and spacecraft speed."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, computing Earth-frame travel time, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Apply time dilation using the Lorentz factor to convert Earth-frame travel time into astronaut's proper time
    cot_instruction_5 = (
        "Sub-task 5: Apply time dilation using the Lorentz factor to convert the Earth-frame travel time into the astronauts proper time "
        "(the time experienced by the astronaut)."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, applying time dilation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 1: Evaluate astronaut survival
    
    # Sub-task 6: Evaluate whether astronaut's proper travel time is less than remaining lifetime
    cot_instruction_6 = (
        "Sub-task 6: Evaluate whether the astronauts proper travel time (from subtask 5) is less than the astronauts remaining lifetime "
        "(150 years average lifetime minus current age 22 years), to determine if the astronaut can survive the journey."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, thinking1, answer1], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating survival, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    # Stage 2: Compare astronaut proper travel time with answer choices
    
    # Sub-task 7: Compare astronaut proper travel time with provided answer choices to select best match
    debate_instruction_7 = (
        "Sub-task 7: Compare the astronauts proper travel time with the provided answer choices (72 years, 77 years, 81 years, or astronaut dies before arrival) "
        "to identify the closest approximate travel time and select the best matching answer."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing travel time with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the best matching answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final answer decision, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer