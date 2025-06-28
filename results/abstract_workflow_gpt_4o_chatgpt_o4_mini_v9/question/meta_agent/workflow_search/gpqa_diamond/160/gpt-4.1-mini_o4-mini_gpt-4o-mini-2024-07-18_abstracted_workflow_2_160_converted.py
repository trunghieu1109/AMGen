async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Define Mean Free Path Concepts

    # Sub-task 1: Analyze physical setup and conditions
    cot_instruction_1 = (
        "Sub-task 1: Analyze the physical setup and conditions described in the query, including the HRTEM operating at 1000 kV, "
        "the ultra-high vacuum (< 10^-9 Torr), and residual gas molecules detected by the mass spectrometer. Identify key parameters such as pressure, temperature, "
        "and sample compartment volume relevant to mean free path calculations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Define and explain mean free path λ1
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, define and explain the concept of mean free path (λ) for gas molecules in an ultra-high vacuum environment, "
        "specifically λ1 as determined by Mike based on pressure, temperature, and volume before electron beam initiation."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, defining mean free path λ1, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Explain effect of electron beam on mean free path λ2
    cot_instruction_3 = (
        "Sub-task 3: Explain the effect of initiating the electron beam on the interaction between electrons and residual gas molecules, "
        "focusing on how electron scattering can influence the effective mean free path (λ2) observed during electron microscopy."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, explaining electron beam effect on mean free path, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Compare and Evaluate Mean Free Paths

    # Sub-task 4: Compare λ1 and λ2 under same conditions
    cot_instruction_4 = (
        "Sub-task 4: Compare and contrast the two mean free paths λ1 (gas molecule mean free path without electron beam) and λ2 (mean free path considering electron scattering) "
        "under the same temperature and vacuum conditions, analyzing physical principles that could cause λ2 to differ from λ1."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing λ1 and λ2, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate possible relationships between λ1 and λ2 with debate
    debate_roles = ["Pro-λ2<λ1", "Pro-λ2=λ1", "Pro-λ1<λ2<1.22λ1", "Pro-λ2>=1.22λ1"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, evaluate the possible relationships between λ1 and λ2 "
        "based on known electron scattering theory and vacuum physics, and determine which of the provided choices best fits the scenario described."
    )
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating λ1 and λ2 relationship, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the relationship between λ2 and λ1.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding λ2 and λ1 relationship, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Conclude and justify the selected relationship
    cot_reflect_instruction_6 = (
        "Sub-task 6: Conclude the reasoning by justifying the selected relationship between λ2 and λ1, "
        "considering the physical implications of electron scattering on mean free path in an ultra-high vacuum environment during electron microscopy."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, concluding and justifying λ2 and λ1 relationship, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
