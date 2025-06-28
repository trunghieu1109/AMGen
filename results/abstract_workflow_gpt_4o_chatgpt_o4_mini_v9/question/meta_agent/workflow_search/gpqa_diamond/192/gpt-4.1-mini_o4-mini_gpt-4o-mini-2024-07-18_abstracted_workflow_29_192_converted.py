async def forward_192(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Understand and transform the relationship between parallax and distance
    # Sub-task 1: Understand that parallax plx = 1/r (inverse proportionality)
    cot_instruction_1 = (
        "Sub-task 1: Understand the relationship between parallax (plx) and distance (r), "
        "specifically that parallax is inversely proportional to distance (plx = 1/r). "
        "Explain why this is foundational to transform the star count dependence on parallax into dependence on distance."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding parallax-distance relation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Express star count ~ 1/plx^5 in terms of distance r by substituting plx = 1/r
    cot_instruction_2 = (
        "Sub-task 2: Using the relation plx = 1/r from Sub-task 1, express the star count dependence "
        "on parallax (number of stars ~ 1/plx^5) in terms of distance r. Derive the functional form of star count with respect to r."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, expressing star count in terms of distance, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Interpret the meaning of number of stars per unit distance range r
    cot_instruction_3 = (
        "Sub-task 3: Interpret the meaning of the number of stars in that region of the sky per unit range of distance r, "
        "relate it to the derived expression from Sub-task 2, and ensure the correct variable and units are considered (stars per unit distance interval)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final]]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, interpreting star count per unit distance, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                                "Review the interpretation of star count per unit distance and provide limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on interpretation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining interpretation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Compare derived functional form with given multiple-choice options
    debate_instruction_4 = (
        "Sub-task 4: Compare the derived functional form of the number of stars per unit distance range r "
        "with the given multiple-choice options (~ r^2, ~ r^3, ~ r^4, ~ r^5) and identify which option matches the derived relationship. "
        "Provide reasoning for the choice."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                input_infos_4 = [taskInfo, thinking3, answer3]
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing derived form with choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                     "Sub-task 4: Make final decision on which choice matches the derived star count dependence on distance.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent on choice selection, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
