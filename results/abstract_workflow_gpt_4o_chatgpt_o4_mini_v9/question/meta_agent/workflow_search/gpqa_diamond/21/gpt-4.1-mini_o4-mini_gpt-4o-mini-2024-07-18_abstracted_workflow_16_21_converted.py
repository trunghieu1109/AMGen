async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Understanding and Clarification of Terms (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Understand and clarify the meaning of the terms 'thermodynamically' and 'kinetically' "
        "in the context of redox reactions, specifically for oxygen in electrochemistry, to set the foundation for analyzing the question."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, clarifying terms thermodynamically and kinetically, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Sub-task 2: Thermodynamic behavior of oxygen in basic solutions (Self-Consistency CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the understanding from Sub-task 1, identify and explain the thermodynamic behavior of oxygen as an oxidant in basic solutions, "
        "focusing on whether it is considered a stronger or weaker oxidant under these conditions."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing thermodynamic behavior in basic solution, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Sub-task 3: Kinetic behavior of oxygen in acidic solutions (Self-Consistency CoT)
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on the understanding from Sub-task 1, identify and explain the kinetic behavior of oxygen reacting in acidic solutions, "
        "focusing on whether the reaction proceeds faster or slower under these conditions."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing kinetic behavior in acidic solution, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Sub-task 4: Compare and contrast thermodynamic and kinetic characteristics (Reflexion)
    cot_reflect_instruction_4 = (
        "Sub-task 4: Compare and contrast the thermodynamic and kinetic characteristics of oxygen in basic and acidic solutions respectively, "
        "to determine the correct combination of descriptors (weaker/stronger and faster/slower) that fits the incomplete test statement."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2_final, answermapping_2[answer2_final], thinking3_final, answermapping_3[answer3_final]]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, comparing thermodynamic and kinetic characteristics, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Please review the comparison and determine if the combination of descriptors is correct and justified.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining comparison, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Sub-task 5: Match the derived combination with multiple-choice options (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, match the derived correct combination of thermodynamic and kinetic descriptors "
        "with the provided multiple-choice options to select the correct answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, matching combination with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
