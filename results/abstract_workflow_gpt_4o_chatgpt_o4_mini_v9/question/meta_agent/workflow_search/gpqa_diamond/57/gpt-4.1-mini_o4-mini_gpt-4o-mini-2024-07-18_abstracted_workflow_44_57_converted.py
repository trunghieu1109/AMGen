async def forward_57(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and classify the four given physical theories
    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the four physical theories: Superstring Theory, Quantum Electrodynamics, "
        "Quantum Chromodynamics, and Classical Electrodynamics by identifying their fundamental nature, domain of applicability, "
        "and typical energy scale considerations relevant to regularization."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing and classifying four physical theories, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Evaluate the need for regularization at high energies for each theory
    # Use Self-Consistency Chain-of-Thought (SC-CoT) for QED and QCD due to complexity
    # Use Chain-of-Thought for Classical Electrodynamics and Superstring Theory

    # Sub-task 2: Evaluate regularization need in Quantum Electrodynamics (QED) with SC-CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Evaluate the need for regularization at high energies in Quantum Electrodynamics by reviewing its renormalization properties, "
        "known divergences, and behavior at high energy scales, based on Sub-task 1 output."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, evaluating QED regularization, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for QED
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Evaluate regularization need in Quantum Chromodynamics (QCD) with SC-CoT
    cot_sc_instruction_3 = (
        "Sub-task 3: Evaluate the need for regularization at high energies in Quantum Chromodynamics by reviewing its renormalization properties, "
        "asymptotic freedom, and divergences, based on Sub-task 1 output."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating QCD regularization, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Evaluate regularization need in Classical Electrodynamics (CED) with CoT
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the need for regularization at high energies in Classical Electrodynamics by analyzing its classical nature "
        "and whether quantum-level regularization concepts apply, based on Sub-task 1 output."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating Classical Electrodynamics regularization, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Evaluate regularization need in Superstring Theory (SST) with CoT
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the need for regularization at high energies in Superstring Theory by analyzing its fundamental framework "
        "and whether it inherently avoids divergences requiring regularization, based on Sub-task 1 output."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating Superstring Theory regularization, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 2: Compare and classify the four theories based on their evaluated need for regularization
    debate_instruction_6 = (
        "Sub-task 6: Compare and classify the four theories (Superstring Theory, Quantum Electrodynamics, Quantum Chromodynamics, Classical Electrodynamics) "
        "based on their evaluated need for regularization at high energies from Sub-tasks 2 to 5, and identify which theory never requires regularization."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                input_infos_6 = [taskInfo, thinking2_final, answermapping_2[answer2_final], thinking3_final, answermapping_3[answer3_final], thinking4, answer4, thinking5, answer5]
            else:
                input_infos_6 = [taskInfo, thinking2_final, answermapping_2[answer2_final], thinking3_final, answermapping_3[answer3_final], thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing regularization needs, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which physical theory never requires regularization at high energies.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding which theory never requires regularization, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
