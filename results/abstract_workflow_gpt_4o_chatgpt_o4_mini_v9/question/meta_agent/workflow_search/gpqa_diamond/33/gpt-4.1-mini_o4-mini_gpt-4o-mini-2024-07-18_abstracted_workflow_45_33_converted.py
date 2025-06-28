async def forward_33(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Foundational understanding and structural analysis
    # Sub-task 1: Understand Pinacol-Pinacolone rearrangement mechanism with Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Understand the Pinacol-Pinacolone rearrangement reaction mechanism, specifically how vicinal diols react under acidic conditions to form carbocations and cause group shifts. "
        "Explain the mechanism step-by-step to build foundational knowledge necessary for analysis."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding Pinacol rearrangement mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze structures and substituents of the three vicinal diols with Self-Consistency CoT
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the structures and substituents of the three given vicinal diols: "
        "(1) 3-methyl-4-phenylhexane-3,4-diol, (2) 3-(4-hydroxyphenyl)-2-phenylpentane-2,3-diol, and "
        "(3) 1,1,2-tris(4-methoxyphenyl)-2-phenylethane-1,2-diol. Identify hydroxyl group positions and possible carbocation formation sites. "
        "Use multiple reasoning paths to ensure consistency."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing diol structures, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Predict carbocation intermediates and ketone products
    # Sub-task 3: Predict carbocation intermediates and group shifts for each diol with Chain-of-Thought
    cot_instruction_3 = (
        "Sub-task 3: For each diol compound, apply the Pinacol rearrangement mechanism to predict the carbocation intermediate formed after protonation and loss of water. "
        "Determine which group shifts (alkyl or aryl) are most likely based on carbocation stability and migratory aptitude."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, predicting carbocation intermediates and group shifts, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Generate possible ketone products with Chain-of-Thought
    cot_instruction_4 = (
        "Sub-task 4: Generate the possible ketone products for each compound after the rearrangement, considering the identified carbocation intermediates and group shifts. "
        "Write the IUPAC names of these predicted products."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, generating ketone products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Compare predicted products with given choices and select correct choice
    # Sub-task 5: Compare predicted ketone products with given answer choices using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the predicted ketone products for each compound with the four given answer choices (choice1 to choice4). "
        "Evaluate which choice matches the predicted products in terms of structure and nomenclature."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct choice matching the predicted products.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Rank and select the correct choice with justification using Reflexion
    cot_reflect_instruction_6 = (
        "Sub-task 6: Based on the comparison results, rank and select the correct choice that best corresponds to the products of the Pinacol rearrangement for all three compounds. "
        "Provide detailed justification based on mechanistic reasoning and structural analysis."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round if hasattr(self, 'max_round') else 3
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, initial justification, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the justification and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining justification round {i+1}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
