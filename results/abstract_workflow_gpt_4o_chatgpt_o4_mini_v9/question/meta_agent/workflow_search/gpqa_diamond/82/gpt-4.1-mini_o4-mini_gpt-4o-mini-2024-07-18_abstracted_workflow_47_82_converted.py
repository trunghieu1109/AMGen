async def forward_82(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Calculate initial moles, total volume after dilution, and new molarity after dilution
    
    # Sub-task 1: Calculate initial moles of acetic acid before dilution
    cot_instruction_1 = (
        "Sub-task 1: Calculate the initial moles of acetic acid before dilution using the given volume (20.00 cm³) "
        "and molarity (0.05 M). This establishes the starting amount of acid for further calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating initial moles of acetic acid, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Determine total volume after dilution
    cot_instruction_2 = (
        "Sub-task 2: Determine the total volume of the solution after dilution by adding 20.00 cm³ of water to the original 20.00 cm³ acetic acid solution. "
        "This is necessary to find the new concentration of acetic acid after dilution."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total volume after dilution, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Calculate new molarity after dilution using initial moles and total volume
    cot_instruction_3 = (
        "Sub-task 3: Calculate the new molarity of acetic acid after dilution using the initial moles from Sub-task 1 and the total volume from Sub-task 2. "
        "This concentration will be used for titration calculations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating new molarity after dilution, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Calculate moles of NaOH for 25% titration and equivalence point, then pH at 25% titration and equivalence point
    
    # Sub-task 4: Calculate moles of NaOH required for 25% titration
    cot_instruction_4 = (
        "Sub-task 4: Calculate the moles of NaOH required to reach 25% titration of the acetic acid in the diluted solution, "
        "using the new molarity from Sub-task 3. This defines the amount of base added at 25% titration."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating moles of NaOH for 25% titration, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Calculate moles of NaOH required to reach equivalence point
    cot_instruction_5 = (
        "Sub-task 5: Calculate the moles of NaOH required to reach the equivalence point, which equals the initial moles of acetic acid (from Sub-task 1), "
        "since at equivalence all acid is neutralized."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking1, answer1]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating moles of NaOH for equivalence point, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Sub-task 6: Calculate pH at 25% titration using Henderson-Hasselbalch equation
    cot_instruction_6 = (
        "Sub-task 6: Calculate the pH at 25% titration point using the Henderson-Hasselbalch equation. "
        "Use the moles of acetic acid and acetate ion after 25% neutralization, the Ka value (1.85x10^-5), and the total volume after dilution."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking3, answer3, thinking4, answer4]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating pH at 25% titration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    # Sub-task 7: Calculate pH at equivalence point by hydrolysis of acetate ion
    cot_instruction_7 = (
        "Sub-task 7: Calculate the pH at the equivalence point by determining the concentration of acetate ion formed, "
        "using the moles of NaOH at equivalence (Sub-task 5) and total volume after dilution and titration. "
        "Then calculate pOH from hydrolysis of acetate ion and convert to pH."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_7 = [taskInfo, thinking2, answer2, thinking5, answer5]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating pH at equivalence point, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    
    # Stage 3: Final decision making - synthesize pH values at 25% titration and equivalence point
    debate_instruction_8 = (
        "Sub-task 8: Based on the outputs of Sub-tasks 6 and 7, synthesize and finalize the pH values at 25% titration and equivalence point. "
        "Compare with given choices and provide the best matching answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7]
            else:
                input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
            thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing final pH values, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on pH values at 25% titration and equivalence point.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing final pH values, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
