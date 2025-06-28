async def forward_171(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and derive expression for excitation ratio using Boltzmann distribution

    # Sub-task 1: Analyze physical context and significance of excitation levels and energy difference
    cot_instruction_1 = (
        "Sub-task 1: Analyze the physical context of the problem: identify the meaning of excitation levels of iron atoms in the photospheres of two stars, "
        "and understand the significance of the given energy difference (1.38 x 10^(-23) J) and the LTE assumption."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Identify relevant physical law (Boltzmann distribution) relating population ratio to temperature
    cot_instruction_2 = (
        "Sub-task 2: Identify and write down the relevant physical law or equation that relates the population of atoms in different energy levels to temperature under LTE conditions, specifically the Boltzmann distribution."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, identifying physical law, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Express ratio of excited iron atoms in star_1 to star_2 in terms of T1 and T2 using Boltzmann distribution and given energy difference
    cot_instruction_3 = (
        "Sub-task 3: Express the ratio of the number of excited iron atoms in star_1 to star_2 in terms of their effective temperatures (T_1 and T_2) using the Boltzmann distribution and the given energy difference."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing excitation ratio, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Manipulate expression and compare with given choices

    # Sub-task 4: Manipulate expression to isolate ln(2) and simplify equation involving T1 and T2
    cot_instruction_4 = (
        "Sub-task 4: Manipulate the expression from Sub-task 3 to isolate and simplify the equation involving ln(2) (since the excitation ratio is 2) and the temperatures T_1 and T_2, aiming to match the form of the given choices."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, manipulating expression, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Compare derived equation with provided multiple-choice options to identify correct relationship
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, compare the derived equation with the provided multiple-choice options and identify which equation correctly represents the relationship between T_1 and T_2."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing derived equation with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which equation correctly represents the relationship between T_1 and T_2.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct equation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
