async def forward_66(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Understand and restate the physical system and quantum state
    cot_instruction_1 = (
        "Sub-task 1: Understand and restate the physical system and given quantum state: "
        "two electrons each in p orbitals (l1=1, l2=1), combined into total angular momentum state |1,1,2,-1>. "
        "Clarify what the quantum numbers represent and the measurement to be made (L1z and L2z eigenvalues both equal to -ħ)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding physical system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1, Sub-task 2: Identify the uncoupled basis states |l1,m1>|l2,m2>
    cot_instruction_2 = (
        "Sub-task 2: Identify the basis states for the two-electron system in terms of the uncoupled basis |l1,m1>|l2,m2>, "
        "where m1 and m2 are the z-components of the individual orbital angular momenta, each can be -1, 0, or +1."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, identifying uncoupled basis states, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Express |1,1,2,-1> in uncoupled basis using Clebsch-Gordan coefficients
    cot_sc_instruction_3 = (
        "Sub-task 3: Express the coupled state |1,1,2,-1> in terms of the uncoupled basis states |1,m1>|1,m2> "
        "using Clebsch-Gordan coefficients. Find expansion coefficients for all (m1,m2) with m1+m2 = -1."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, expanding coupled state, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (mode) or first if tie
    from collections import Counter
    answer_counts = Counter(possible_answers_3)
    most_common_answer, _ = answer_counts.most_common(1)[0]
    thinking3 = thinkingmapping_3[most_common_answer]
    answer3 = answermapping_3[most_common_answer]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2, Sub-task 4: Identify the uncoupled basis state |1,-1>|1,-1> corresponding to measurement L1z = -ħ, L2z = -ħ
    cot_instruction_4 = (
        "Sub-task 4: Identify the specific uncoupled basis state corresponding to measurement outcome L1z = -ħ and L2z = -ħ, "
        "i.e., |1,m1=-1>|1,m2=-1>."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying measurement basis state, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 3, Sub-task 5: Calculate joint probability of measuring L1z = -ħ and L2z = -ħ
    debate_instruction_5 = (
        "Sub-task 5: Calculate the joint probability of measuring L1z = -ħ and L2z = -ħ by taking the squared magnitude of the Clebsch-Gordan coefficient "
        "corresponding to the |1,-1>|1,-1> component in the expansion of |1,1,2,-1>."
    )
    debate_roles = getattr(self, 'debate_role', ['Proponent', 'Opponent'])
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating joint probability, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the joint probability of measuring L1z = -ħ and L2z = -ħ.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating joint probability, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
