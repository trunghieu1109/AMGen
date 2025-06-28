async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instr1 = "Sub-task 1: Parse the quantum state ψ = 0.5|↑⟩ + (√3/2)|↓⟩ and the operator 10σ_z + 5σ_x, extracting amplitudes a and b and identifying basis states."
    thinking1, answer1 = await cot1([taskInfo], instr1, is_sub_task=True)
    agents.append(f"CoT agent {cot1.id}, parsing state and operator, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    cot2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instr2 = "Sub-task 2: Recall the formulas ⟨σ_z⟩ = |a|^2 - |b|^2 and ⟨σ_x⟩ = 2 Re(a* b) for expectation values in the σ_z basis."
    thinking2, answer2 = await cot2([taskInfo, thinking1, answer1], instr2, is_sub_task=True)
    agents.append(f"CoT agent {cot2.id}, recalling formulas for expectation values, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    cot3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instr3 = "Sub-task 3: Using a=0.5 and b=√3/2, compute ⟨σ_z⟩ = |a|^2 - |b|^2."
    thinking3, answer3 = await cot3([taskInfo, thinking1, answer1, thinking2, answer2], instr3, is_sub_task=True)
    agents.append(f"CoT agent {cot3.id}, computing ⟨σ_z⟩, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    cot4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instr4 = "Sub-task 4: Using a=0.5 and b=√3/2, compute ⟨σ_x⟩ = 2 Re(a* b)."
    thinking4, answer4 = await cot4([taskInfo, thinking1, answer1, thinking2, answer2], instr4, is_sub_task=True)
    agents.append(f"CoT agent {cot4.id}, computing ⟨σ_x⟩, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Multiply ⟨σ_z⟩ by 10 and ⟨σ_x⟩ by 5, then sum to get ⟨10σ_z+5σ_x⟩."
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            inputs5 = [taskInfo, thinking3, answer3, thinking4, answer4]
            if r > 0:
                inputs5 += all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(inputs5, debate_instruction5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, combining expectations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make a final decision on the raw expectation value.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding raw expectation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    cot6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instr6 = "Sub-task 6: Round the raw expectation to one decimal place and select the matching choice from 0.85, 1.65, -1.4, -0.7."
    thinking6, answer6 = await cot6([taskInfo, thinking5, answer5], instr6, is_sub_task=True)
    agents.append(f"CoT agent {cot6.id}, rounding and selecting answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer