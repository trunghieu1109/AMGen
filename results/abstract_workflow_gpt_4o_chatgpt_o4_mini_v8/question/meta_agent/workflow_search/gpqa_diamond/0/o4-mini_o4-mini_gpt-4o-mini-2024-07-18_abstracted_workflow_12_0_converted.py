async def forward_0(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Sub-task 1: Compute ΔE₁ for τ₁ = 10⁻⁹ s
    cot_instruction1 = "Sub-task 1: Given lifetime τ₁ = 10⁻⁹ s and ħ ≈ 6.582×10⁻¹⁶ eV·s, compute the energy uncertainty ΔE₁ using ΔE ≈ ħ/τ. Provide the numerical result in eV."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, computing ΔE₁, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Compute ΔE₂ for τ₂ = 10⁻⁸ s
    cot_instruction2 = "Sub-task 2: Given lifetime τ₂ = 10⁻⁸ s and ħ ≈ 6.582×10⁻¹⁶ eV·s, compute ΔE₂ using ΔE ≈ ħ/τ. Provide the numerical result in eV."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, computing ΔE₂, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine δE_min = max(ΔE₁, ΔE₂)
    cot_instruction3 = "Sub-task 3: Determine the minimum energy separation δE_min = max(ΔE₁, ΔE₂) using the results from sub-tasks 1 and 2."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, determining δE_min, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare candidate energies to δE_min using Self-Consistency CoT
    sc_instruction4 = "Sub-task 4: For each candidate energy [10^-9 eV, 10^-11 eV, 10^-8 eV, 10^-4 eV], compare its value to δE_min and state True if it exceeds the threshold, False otherwise."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinking_map4 = {}
    answer_map4 = {}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, comparing options, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinking_map4[answer4_i.content] = thinking4_i
        answer_map4[answer4_i.content] = answer4_i
    most_common4 = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_map4[most_common4]
    answer4 = answer_map4[most_common4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Identify valid options via Debate
    debate_instruction5 = "Sub-task 5: From the comparison in sub-task 4, list which options satisfy option_value > δE_min."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting valid options, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Provide the final list of valid options.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision5.id}, listing valid options, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Select the smallest valid option
    cot_reflect_instruction6 = "Sub-task 6: From the valid options, select the smallest energy difference that still exceeds δE_min."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_reflect_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, selecting smallest valid, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Present final answer
    cot_instruction7 = "Sub-task 7: Present the final answer: the chosen energy difference option for clear resolution."
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, presenting final answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer